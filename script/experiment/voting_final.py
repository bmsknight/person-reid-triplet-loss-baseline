from reid_functions import *
import shutil
import time
import json

cfg = Config()
TVT, TMO = set_devices((0,))
model = Model(last_conv_stride=cfg.last_conv_stride)
model_w = DataParallel(model)
tri_loss = TripletLoss(margin=cfg.margin)

optimizer = optim.Adam(model.parameters(),
                     lr=cfg.base_lr,
                     weight_decay=cfg.weight_decay)

modules_optims = [model, optimizer]

map_location = (lambda storage, loc: storage)
sd = torch.load(cfg.model_weight_file, map_location=map_location)
load_state_dict(model, sd)

ext = ExtractFeature(model=model_w,TVT=TVT)
root_directory = '/home/bmsknight/triplet/person-reid-triplet-loss-baseline/data/ourdata/Database/Dynamic_Database/'
db_name_list, db_feature_list = extract_DB_features(ext,cfg,root_directory)

querrypath = "/home/bmsknight/triplet/person-reid-triplet-loss-baseline/data/ourdata/Database/query5/"
temppath = "/home/bmsknight/triplet/person-reid-triplet-loss-baseline/data/ourdata/Database/temp/"


start = time.time()
n = 450
previous_frame = 0
attendance = {}

for k in range(20):
    copy_n_frames(n,previous_frame,querrypath,temppath)
    querry_name_list, querry_feature_list = extract_querry_features(ext, cfg, querrypath =temppath)
    q_q_dist = compute_dist(querry_feature_list,querry_feature_list, type='cosine')
    g_g_dist = compute_dist(db_feature_list,db_feature_list, type='cosine') 
    q_g_dist_cos = compute_dist(querry_feature_list,db_feature_list, type='cosine')
    q_g_dist_cos = np.ones_like(q_g_dist_cos) - q_g_dist_cos
    q_q_dist = np.ones_like(q_q_dist) - q_q_dist
    g_g_dist = np.ones_like(g_g_dist) - g_g_dist
    reranked_list = re_ranking(q_g_dist_cos, q_q_dist, g_g_dist)
    querry_name_list = querry_name_list.astype(int)
    voted_dictionary, voted_name_list = vote_for_person(reranked_list,querry_name_list,db_name_list)
#     print k
#     for item in voted_dictionary:
#         print( item, voted_dictionary[item])
    display_dictionary, coordinate_dictionary, centroid_dictionary = find_valid_persons(voted_dictionary,voted_name_list,querry_name_list,n/10)
    
#     for item in display_dictionary:
#         print( item, display_dictionary[item])
    for item in coordinate_dictionary:
        if item in attendance.keys():
            attendance[item] = attendance[item]+1
        else:
            attendance[item] = 1
#         print( item, coordinate_dictionary[item])
        
    with open('centroids.txt', 'w') as file:
        file.write(json.dumps(centroid_dictionary))
    
    shutil.rmtree(temppath[:-1])
    os.mkdir(temppath[:-1])
    previous_frame = previous_frame + n

total_time = time.time()-start

for item in attendance:
    attendance[item] = attendance[item]*100/(k+1)
#     print( item, attendance[item])
        
with open('attendance.txt', 'w') as file:
    file.write(json.dumps(attendance))

print (total_time)        