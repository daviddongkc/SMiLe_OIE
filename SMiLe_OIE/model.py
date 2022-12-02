from typing import Dict, List, Optional, Any, Union
from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout, Embedding, LogSigmoid, LogSoftmax
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode, batched_index_select
from allennlp.training.metrics import Metric
import torch.nn as nn

@Model.register("oie_bert_multi_view")
class SMiLe_OIE(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: Union[str, BertModel],
                 embedding_dropout: float = 0.0,
                 dependency_label_dim: int = 400,
                 constituency_label_dim: int = 400,
                 use_graph: bool = True,
                 hyper_div: float = 0.02,
                 hyper_c1: float = 0.05,
                 hyper_c2: float = 0.05,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 tuple_metric: Metric = None) -> None:
        super(SMiLe_OIE, self).__init__(vocab, regularizer)

        self.bert_model = BertModel.from_pretrained(bert_model)
        self.num_classes = self.vocab.get_vocab_size("labels")
        # self._use_graph = use_graph
        self.hyper_div = hyper_div
        self.hyper_c1 = hyper_c1
        self.hyper_c2 = hyper_c2
        self.num_dep_labels = self.vocab.get_vocab_size("dependency_labels")
        self.num_const_labels = self.vocab.get_vocab_size("constituency_labels")
        self.dep_gcn = DepGCN(self.num_dep_labels, dependency_label_dim, self.bert_model.config.hidden_size, self.bert_model.config.hidden_size)
        self.const_gcn = ConstGCN(self.num_const_labels, constituency_label_dim, self.bert_model.config.hidden_size, self.bert_model.config.hidden_size)
        self.tag_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_classes)
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self._token_based_metric = tuple_metric
        self.LogSoftmax = LogSoftmax()
        initializer(self)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                verb_indicator: torch.Tensor,
                dep_nodes: Dict[str, torch.Tensor],
                dep_edges: Dict[str, torch.Tensor],
                const_nodes: Dict[str, torch.Tensor],
                const_edges: Dict[str, torch.Tensor],
                dep_1_start: torch.Tensor,
                dep_1_end: torch.Tensor,
                const_1_start: torch.Tensor,
                const_1_end: torch.Tensor,
                dep_3_start: torch.Tensor,
                dep_3_const_end: torch.Tensor,
                const_3_start: torch.Tensor,
                const_3_dep_end: torch.Tensor,
                metadata: List[Any],
                tags: torch.LongTensor = None,
                optimizer=None):
        mask = get_text_field_mask(tokens)
        bert_embeddings, _ = self.bert_model(input_ids=tokens["tokens"], token_type_ids=verb_indicator, attention_mask=mask, output_all_encoded_layers=False)
        embedded_text_input = self.embedding_dropout(bert_embeddings)
        # get 2 sets of representation from dependency graph and constituency graph.
        embedded_dep = self.dep_gcn(embedded_text_input, dep_edges, dep_nodes['dep_tags'])
        embedded_const = self.const_gcn(embedded_text_input, const_edges, const_nodes['const_tags'])
        embedded_text_input = (embedded_dep + embedded_const) / 2
        batch_size, sequence_length, _ = embedded_text_input.size()
        logits = self.tag_projection_layer(embedded_text_input)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size, sequence_length, self.num_classes])
        output_dict = {"logits": logits, "class_probabilities": class_probabilities, "mask": mask}

        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits, tags, mask, label_smoothing=self._label_smoothing)
            loss_multi_view = self.multi_view_loss(embedded_dep, embedded_const, dep_1_start, dep_1_end, const_1_start,
                                                   const_1_end, dep_3_start, dep_3_const_end, const_3_start, const_3_dep_end)
            if self.hyper_div > 0: loss += self.hyper_div * loss_multi_view['div']
            if self.hyper_c1 > 0: loss += self.hyper_c1 * loss_multi_view['c1']
            if self.hyper_c2 > 0: loss += self.hyper_c2 * loss_multi_view['c2']
            output_dict["loss"] = loss

        # We add in the offsets here so we can compute the un-wordpieced tags.
        words, verbs, offsets = zip(*[(x["words"], x["verb"], x["offsets"]) for x in metadata])
        output_dict["words"] = list(words)
        output_dict["verb"] = list(verbs)
        output_dict["wordpiece_offsets"] = list(offsets)

        if metadata[0]['validation']:
            output_dict = self.decode(output_dict)
            # think about how to get confidence score
            predicates_index = [x["verb_index"] for x in metadata]
            self._token_based_metric(output_dict["words"], output_dict["tags"], predicates_index, output_dict["tag_probs"])
        return output_dict

    def multi_view_loss(self, embedded_dep, embedded_const, dep_1_start, dep_1_end, const_1_start, const_1_end,
                        dep_3_start, dep_3_const_end, const_3_start, const_3_dep_end):
        bs = embedded_dep.size(0)
        seq_size = embedded_dep.size(1)
        embed_dim = embedded_dep.size(2)
        loss = {}
        # 1. intra-view diversity score
        dep_size = dep_1_start.size(1)
        node_centre_embed = batched_index_select(embedded_dep, dep_1_start.long()).view(bs*dep_size, embed_dim, -1)
        node_context_embed = batched_index_select(embedded_dep, dep_1_end.long()).view(bs*dep_size, -1, embed_dim)
        score_intra_dep = torch.bmm(node_context_embed, node_centre_embed).view(bs, dep_size, 1, 1)
        loss_dep = self.LogSoftmax(score_intra_dep).squeeze().mean()
        const_size = const_1_start.size(1)
        node_centre_embed = batched_index_select(embedded_const, const_1_start.long()).view(bs * const_size, embed_dim, -1)
        node_context_embed = batched_index_select(embedded_const, const_1_end.long()).view(bs * const_size, -1, embed_dim)
        score_intra_const = torch.bmm(node_context_embed, node_centre_embed).view(bs, const_size, 1, 1)
        loss_const = self.LogSoftmax(score_intra_const).squeeze().mean()
        loss['div'] = -(loss_dep + loss_const)/2
        # 2. inter-view intra-node 1st order collaboration score
        node_centre_embed = embedded_dep.view(bs * seq_size, embed_dim, -1)
        node_context_embed = embedded_const.view(bs * seq_size, -1, embed_dim)
        score_inter_dep_const = torch.bmm(node_context_embed, node_centre_embed).view(bs, seq_size, 1, 1)
        loss_inter_dep_const = self.LogSoftmax(score_inter_dep_const).squeeze().mean()
        loss['c1'] = -loss_inter_dep_const
        # 3. inter-view inter-node 2nd order collaboration score
        dep_3_size = dep_3_start.size(1)
        node_centre_embed = batched_index_select(embedded_dep, dep_3_start.long()).view(bs * dep_3_size, embed_dim, -1)
        node_context_embed = batched_index_select(embedded_const, dep_3_const_end.long()).view(bs * dep_3_size, -1, embed_dim)
        score_inter_dep = torch.bmm(node_context_embed, node_centre_embed).view(bs, dep_3_size, 1, 1)
        loss_inter_dep = self.LogSoftmax(score_inter_dep).squeeze().mean()
        const_3_size = const_3_start.size(1)
        node_centre_embed = batched_index_select(embedded_const, const_3_start.long()).view(bs * const_3_size, embed_dim, -1)
        node_context_embed = batched_index_select(embedded_dep, const_3_dep_end.long()).view(bs * const_3_size, -1, embed_dim)
        score_inter_const = torch.bmm(node_context_embed, node_centre_embed).view(bs, const_3_size, 1, 1)
        loss_inter_const = self.LogSoftmax(score_inter_const).squeeze().mean()
        loss['c2'] = -(loss_inter_dep + loss_inter_const)/2
        return loss


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        wordpiece_tags, word_tags, wordpiece_tag_probs, word_tag_probs, tag_ids = [], [], [], [], []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        start_transitions = self.get_start_transitions()
        # **************** Different ********************
        # We add in the offsets here so we can compute the un-wordpieced tags.
        for predictions, length, offsets in zip(predictions_list, sequence_lengths, output_dict["wordpiece_offsets"]):
            max_likelihood_sequence, _ = viterbi_decode(predictions[:length], transition_matrix, allowed_start_transitions=start_transitions)
            # get predicted tags from index
            tags = [self.vocab.get_token_from_index(x, namespace="labels") for x in max_likelihood_sequence]
            wordpiece_tags.append(tags)
            word_tags.append([tags[i] for i in offsets])
            # get the confidence score of predicted tags
            tag_probs = [float(predictions[i][j]) for i, j in enumerate(max_likelihood_sequence)]
            wordpiece_tag_probs.append(tag_probs)
            word_tag_probs.append([tag_probs[i] for i in offsets])
            tag_ids.append([max_likelihood_sequence[i] for i in offsets])

        output_dict['wordpiece_tags'] = wordpiece_tags
        output_dict['tags'] = word_tags
        output_dict['wordpiece_tag_probs'] = wordpiece_tag_probs
        output_dict['tag_probs'] = word_tag_probs
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False):
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics

    def get_viterbi_pairwise_potentials(self):
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])
        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

    def get_start_transitions(self):
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        start_transitions = torch.zeros(num_labels)
        for i, label in all_labels.items():
            if label[0] == "I":
                start_transitions[i] = float("-inf")
        return start_transitions

class DepGCN(nn.Module):
    """
    Label-aware Dependency Convolutional Neural Network Layer
    """
    def __init__(self, dep_num, dep_dim, in_features, out_features):
        super(DepGCN, self).__init__()
        self.dep_dim = dep_dim
        self.in_features = in_features
        self.out_features = out_features
        self.dep_embedding = nn.Embedding(dep_num, dep_dim, padding_idx=0)
        self.dep_attn = nn.Linear(dep_dim + in_features, out_features)
        self.dep_fc = nn.Linear(dep_dim, out_features)
        self.relu = nn.ReLU()

    def forward(self, text, dep_mat, dep_labels):
        dep_label_embed = self.dep_embedding(dep_labels)
        batch_size, seq_len, feat_dim = text.shape
        val_dep = dep_label_embed.unsqueeze(dim=2)
        val_dep = val_dep.repeat(1, 1, seq_len, 1)
        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, seq_len, 1)
        val_sum = torch.cat([val_us, val_dep], dim=-1)
        r = self.dep_attn(val_sum)
        p = torch.sum(r, dim=-1)
        mask = (dep_mat == 0).float() * (-1e30)
        p = p + mask
        p = torch.softmax(p, dim=2)
        p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim)
        output = val_us + self.dep_fc(val_dep)
        output = torch.mul(p_us, output)
        output_sum = torch.sum(output, dim=2)
        output_sum = self.relu(output_sum)
        return output_sum



class ConstGCN(nn.Module):
    """
    Label-aware Constituency Convolutional Neural Network Layer
    """
    def __init__(self, const_num, const_dim, in_features, out_features):
        super(ConstGCN, self).__init__()
        self.const_num = const_num
        self.in_features = in_features
        self.out_features = out_features
        self.const_embedding = nn.Embedding(const_num, const_dim, padding_idx=0)
        self.const_attn = nn.Linear(const_dim + in_features, out_features)
        self.const_fc = nn.Linear(const_dim, out_features)
        self.relu = nn.ReLU()

    def forward(self, text, const_mat, const_labels):
        const_label_embed = self.const_embedding(const_labels)
        const_label_embed = torch.mean(const_label_embed, 2)
        batch_size, seq_len, feat_dim = text.shape
        val_dep = const_label_embed.unsqueeze(dim=2)
        val_dep = val_dep.repeat(1, 1, seq_len, 1)
        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, seq_len, 1)
        val_sum = torch.cat([val_us, val_dep], dim=-1)
        r = self.const_attn(val_sum)
        p = torch.sum(r, dim=-1)
        mask = (const_mat == 0).float() * (-1e30)
        p = p + mask
        p = torch.softmax(p, dim=2)
        p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim)
        output = val_us + self.const_fc(val_dep)
        output = torch.mul(p_us, output)
        output_sum = torch.sum(output, dim=2)
        output_sum = self.relu(output_sum)
        return output_sum
