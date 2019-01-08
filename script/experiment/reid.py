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
db_name_list, db_feature_list = extract_DB_features(ext,root_directory)

querrypath = "/home/bmsknight/triplet/person-reid-triplet-loss-baseline/data/ourdata/Database/query3/"
for filename in sorted(os.listdir(querrypath)):
    filepath = querrypath+filename
    person = findPerson(filepath, ext, db_name_list, db_feature_list)
    
    print(filename, person)

