from reid_functions import *

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

querrypath = "/home/bmsknight/triplet/person-reid-triplet-loss-baseline/data/ourdata/Database/query3/"
# for filename in sorted(os.listdir(querrypath)):
#     filepath = querrypath+filename
#     person = findPerson(filepath, ext, cfg, db_name_list, db_feature_list)
    
#     print(filename, person)

querry_name_list, querry_feature_list = extract_querry_features(ext, cfg, querrypath = "/home/bmsknight/triplet/person-reid-triplet-loss-baseline/data/ourdata/Database/query3/")

q_q_dist = compute_dist(querry_feature_list,querry_feature_list, type='cosine')
g_g_dist = compute_dist(db_feature_list,db_feature_list, type='cosine') 
q_g_dist_cos = compute_dist(querry_feature_list,db_feature_list, type='cosine')
q_g_dist_cos = np.ones_like(q_g_dist_cos) - q_g_dist_cos
q_q_dist = np.ones_like(q_q_dist) - q_q_dist
g_g_dist = np.ones_like(g_g_dist) - g_g_dist

reranked_list = re_ranking(q_g_dist_cos, q_q_dist, g_g_dist)

count =0
for i in range(len(querry_name_list)):
    rr_pred = db_name_list[np.argmin(reranked_list[i,:])]
    cos_pred = db_name_list[np.argmin(q_g_dist_cos[i,:])]
    
    if rr_pred != cos_pred:
        print(querry_name_list[i], cos_pred, rr_pred)
        count +=1

print(count)