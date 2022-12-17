from BNReasoner import BNReasoner
import pandas as pd

def test_factor_multiplication():
    test_network_path = "testing/lecture_example.BIFXML"
    test_network_bn = BNReasoner(test_network_path)
    test_network_cpts = test_network_bn.bn.get_all_cpts()
    test_factor_multiplication = test_network_bn.factor_multiplication(test_network_cpts["Rain?"], test_network_cpts["Winter?"])
    test_factor_multiplication = test_factor_multiplication.round(2)
    expected_output_df = pd.DataFrame({"Winter?": [False, False, True, True], "Rain?": [False, True, False, True], "p": [0.36, 0.04, 0.12, 0.48]})
    assert test_factor_multiplication.equals(expected_output_df)

def test_network_prune():
    test_network_path = "testing/lecture_example.BIFXML"
    test_network_bn = BNReasoner(test_network_path)
    PrunedGraph = test_network_bn.network_prune(test_network_bn.bn, ["Sprinkler"],["Winter?"])
    assert PrunedGraph.get_all_variables() == ["Winter?"]
    print("test_network_prune passed")

def test_d_separation():
    test_network_path = "testing/lecture_example.BIFXML"
    test_network_bn = BNReasoner(test_network_path)
    assert test_network_bn.d_separation(["Rain?"], ["Wet Grass?"], ["Sprinkler?"]) == True
    print("test_d_separation passed")

def test_independence():
    test_network_path = "testing/lecture_example.BIFXML"
    test_network_bn = BNReasoner(test_network_path)
    assert test_network_bn.independence(["Rain?"], ["Wet Grass?"], ["Sprinkler?"]) == True
    print("test_independence passed")

def test_calculate_marginalisation():
    test_network_path = "testing/lecture_example.BIFXML"
    test_network_bn = BNReasoner(test_network_path)
    df = test_network_bn.bn.get_all_cpts()["Slippery Road?"]
    df_out = test_network_bn.calculate_marginalisation("Rain?",df)
    expected_out = pd.DataFrame({'Slippery Road?': [False, True], 'p': [1.3, 0.7]})
    assert df_out.equals(expected_out)
    print("test_calculate_marginalisation passed")

def test_maxing_out():
    test_network_path = "testing/lecture_example.BIFXML"
    test_network_bn = BNReasoner(test_network_path)
    df = test_network_bn.bn.get_all_cpts()["Slippery Road?"]
    df_out = test_network_bn.maxing_out("Rain?",df)
    #print(df_out)
    expected_out = pd.DataFrame({'Slippery Road?': [False, True],
                                 "inst_Rain?": [False, True],
                                 'p': [1.0, 0.7]})
    assert df_out.equals(expected_out)
    print("test_maxing_out passed")

def test_ordering():
    test_network_path = "testing/lecture_example.BIFXML"
    test_network_bn = BNReasoner(test_network_path)
    mf_ordering = test_network_bn.ordering(ordering_heuristics="min_fill")

    assert mf_ordering == ['Winter?', 'Sprinkler?', "Rain?", 'Wet Grass?', 'Slippery Road?']
    print("test ordering min_fill passed")

    md_ordering = test_network_bn.ordering(ordering_heuristics="min_degree")
    assert md_ordering == ['Slippery Road?','Winter?', 'Sprinkler?', "Rain?", 'Wet Grass?']

    print("test ordering min_degree passed")


def test_variable_elimination():
    test_network_path = "testing/lecture_example.BIFXML"
    test_network_bn = BNReasoner(test_network_path)
    test_network_cpts = test_network_bn.bn.get_all_cpts()
    elim_out = test_network_bn.variable_elimination(test_network_cpts, ordering_set=["Winter?", 'Wet Grass?', 'Slippery Road?', "Rain?"])
    expected_out = pd.DataFrame({'Sprinkler?': [False, True],  'p': [0.58, 0.42]})
    elim_out = list(elim_out.values())[0].round(2)
    assert expected_out.equals(elim_out)
    print("test_variable_elimination passed")


def test_marginal_distribution():
    test_network_path = "testing/lecture_example.BIFXML"
    test_network_bn = BNReasoner(test_network_path)
    test_marginal_df = test_network_bn.marginal_distribution(["Rain?"])
    expected_out = pd.DataFrame({'Rain?': [False, True], 'p': [0.48, 0.52]})
    test_marginal_df = test_marginal_df.round(2)
    assert test_marginal_df.equals(expected_out)
    print("test_marginal_distribution passed")


def test_maximum_a_posteriori():
    test_network_path = "testing/lecture_example.BIFXML"
    test_network_bn = BNReasoner(test_network_path)
    test_map_df = test_network_bn.maximum_a_posteriori(["Rain?"], { "Wet Grass?": True})
    expected_out = pd.DataFrame({'inst_Rain?': [True], 'p': [0.4349]})
    test_map_df = test_map_df.round(4).reset_index(drop=True)
    assert test_map_df.equals(expected_out)
    print("test_maximum_a_posteriori passed")


def test_most_probable_explanation():
    test_network_path = "testing/lecture_example.BIFXML"
    test_network_bn = BNReasoner(test_network_path)
    test_mpe_df = test_network_bn.most_probable_explanation({ "Wet Grass?": True, "Slippery Road?": True})
    test_mpe_df = test_mpe_df.round(5).reset_index(drop=True)
    expected_out = pd.DataFrame({ 'Winter?': [True],'Sprinkler?': [False],'Rain?': [True],  'Slippery Road?': [True], 'inst_Wet Grass?':[True], 'p': [0.21504]})
    assert test_mpe_df.equals(expected_out)
    print("test_most_probable_explanation passed")


test_factor_multiplication()
test_network_prune()
test_d_separation()
test_independence()
test_calculate_marginalisation()
test_maxing_out()
test_ordering()
test_variable_elimination()
test_marginal_distribution()
test_maximum_a_posteriori()
test_most_probable_explanation()