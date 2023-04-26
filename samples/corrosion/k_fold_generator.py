# %%
import os, json, random

# %%
def k_fold_via_generator(src_file, tar_fold, k_fold):
    train_data = []
    val_data = []
    with open(src_file, 'r') as f:
        src_json = json.load(f)

    img_keys = list(src_json.keys())
    random.shuffle(img_keys)
    val_img_num = int(len(img_keys) / k_fold)

    start = 0
    while start < val_img_num * (k_fold - 1):
        val_data.append(set(img_keys[start: start+val_img_num]))
        start += val_img_num
    val_data.append(img_keys[start:])

    for item in val_data:
        train_data.append(set(item) ^ set(img_keys))

    index = 1
    for train_keys, val_keys in zip(train_data, val_data):
        train_json = {}
        val_json = {}
        for item in train_keys:
            train_json.update({item: src_json[item]})
        for item in val_keys:
            val_json.update({item: src_json[item]})

        # with open(os.path.join(tar_fold, "{}_fold_train_group_{}.json".format(k_fold, index)), 'w') as f:
        #     json.dump(train_json, f)
        # with open(os.path.join(tar_fold, "{}_fold_val_group_{}.json".format(k_fold, index)), 'w') as f:
        #     json.dump(val_json, f)
        f1 = open(os.path.join(tar_fold, "{}_fold_train_group_{}.json".format(k_fold, index)), 'w')
        f2 = open(os.path.join(tar_fold, "{}_fold_val_group_{}.json".format(k_fold, index)), 'w')

        json.dump(train_json, f1)
        json.dump(val_json, f2)
        f1.close()
        f2.close()

        index += 1

def train_val_split_via_generator(src_file, tar_fold, percent):
    """

    Args:
        src_file:
        tar_fold:
        percent: len(train)/len(all)

    Returns:

    """
    train_json = {}
    val_json = {}
    with open(src_file, 'r') as f:
        src_json = json.load(f)

    img_keys = list(src_json.keys())
    random.shuffle(img_keys)
    train_data_len = int(len(img_keys) * percent)
    for key in img_keys[:train_data_len]:
        train_json.update({key: src_json[key]})
    for key in img_keys[train_data_len:]:
        val_json.update({key: src_json[key]})
    with open(os.path.join(tar_fold, "train.json"), 'w', encoding="utf-8") as f:
        json.dump(train_json, f)
    with open(os.path.join(tar_fold, "val.json"), 'w', encoding="utf-8") as f:
        json.dump(val_json, f)

if __name__ == "__main__":

    src_file = "./via_region_data.json"
    tar_fold = "./"
    k_fold = 1
    # k_fold_via_generator(src_file, tar_file, k_fold)
    train_val_split_via_generator(src_file, tar_fold, 0.7)

