# create dataset for LSOIE-wiki testing
python convert_conll_to_onenotes.py --inp lsoie_wiki_test.conll --out lsoie_wiki_test.gold_conll
python json_conversion.py --inp lsoie_wiki_test.gold_conll --out lsoie_wiki_test_tree.json
python graph_conversion.py --inp lsoie_wiki_test_tree.json --out ../lsoie_wiki_test.json

# create dataset for LSOIE-wiki training
python convert_conll_to_onenotes.py --inp lsoie_wiki_train.conll --out lsoie_wiki_train.gold_conll
python json_conversion.py --inp lsoie_wiki_train.gold_conll --out lsoie_wiki_train_tree.json
python graph_conversion.py --inp lsoie_wiki_train_tree.json --out ../lsoie_wiki_train.json

# create dataset for LSOIE-science testing
python convert_conll_to_onenotes.py --inp lsoie_science_test.conll --out lsoie_science_test.gold_conll
python json_conversion.py --inp lsoie_science_test.gold_conll --out lsoie_sci_test_tree.json
python graph_conversion.py --inp lsoie_sci_test_tree.json --out ../lsoie_sci_test.json

# create dataset for LSOIE-science training
python convert_conll_to_onenotes.py --inp lsoie_science_train.conll --out lsoie_science_train.gold_conll
python json_conversion.py --inp lsoie_science_train.gold_conll --out lsoie_sci_train_tree.json
python graph_conversion.py --inp lsoie_sci_train_tree.json --out ../lsoie_sci_train.json