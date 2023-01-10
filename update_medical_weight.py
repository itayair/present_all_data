import pickle
import DAG.DAG_utils as DAG_utils

topic_objects = pickle.load(open("results_disease/diabetes/topic_object_lst.p", "rb"))
global_index_to_similar_longest_np = pickle.load(open("results_disease/diabetes/global_index_to_similar_longest_np.p", "rb"))
dict_span_to_rank = pickle.load(open("results_disease/diabetes/dict_span_to_rank.p", "rb"))
global_dict_label_to_object = pickle.load(open("results_disease/diabetes/global_dict_label_to_object.p", "rb"))
DAG_utils.initialize_nodes_weighted_average_vector(topic_objects, global_index_to_similar_longest_np)
pickle.dump(topic_objects, open("results_disease/diabetes/topic_object_lst.p", "wb"))
pickle.dump(global_index_to_similar_longest_np,
            open("results_disease/diabetes/global_index_to_similar_longest_np.p", "wb"))
pickle.dump(dict_span_to_rank, open("results_disease/diabetes/dict_span_to_rank.p", "wb"))
pickle.dump(global_dict_label_to_object, open("results_disease/diabetes/global_dict_label_to_object.p", "wb"))
