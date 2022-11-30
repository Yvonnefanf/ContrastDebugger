from timevis_backend.utils import *

timevis = initialize_backend('/home/yifan/dataset/noisy/pairflip/cifar10/0')
print("timeve",timevis)

embedding_2d, grid, decision_view, label_name_dict, label_color_list, label_list, max_iter, training_data_index, \
testing_data_index, eval_new, prediction_list, selected_points, properties = update_epoch_projection(timevis, 1, {})

# print("embedding_2d",embedding_2d)
