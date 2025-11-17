# train.py
import os, cv2, numpy as np
from glob import glob
from tqdm import tqdm
from uwie import UWIE
from morph import morph_process
from features import make_detector, extract_descriptors, build_vocabulary, compute_bovw_hist
from model import build_patternnet
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_DIR = 'data_fish'    # folder with subfolders per class
VOCAB_K = 128             # number of visual words (reduce to 64/128 if slow)
IMG_SIZE = (64,64)

def load_image_paths(data_dir):
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))]
    paths = []
    labels = []
    for cls in classes:
        files = glob(os.path.join(data_dir,cls,'*.*'))
        for f in files:
            paths.append(f)
            labels.append(cls)
    return paths, labels

def build_bovw_for_dataset(paths):
    det = make_detector()
    desc_list = []
    for p in tqdm(paths, desc='Extract descriptors'):
        img = cv2.imread(p)
        if img is None: continue
        e = UWIE(img)
        crop, _ = morph_process(e)
        des = extract_descriptors(crop, det)
        desc_list.append(des)
    kmeans = build_vocabulary(desc_list, k=VOCAB_K, save_path='kmeans.pkl')
    # build hist for each image
    H = []
    for des in tqdm(desc_list, desc='Compute hist'):
        H.append(compute_bovw_hist(des, kmeans))
    X = np.vstack(H)
    return X, kmeans

def train_bovw_classifier(X, y):
    # run firefly selection (optional)
    from firefly import firefly_feature_select
    mask, score = firefly_feature_select(X, y, n_fireflies=12, max_iter=10)
    print("Firefly best score:", score)
    Xs = X[:, mask==1]
    # train final classifier
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf.fit(Xs, y)
    # save
    import pickle
    pickle.dump({'clf':clf, 'mask':mask}, open('bovw_model.pkl','wb'))

def train_cnn(paths, labels):
    import tensorflow as tf
    X = []
    y = []
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    for p,label in tqdm(zip(paths, y_enc), total=len(paths), desc='Load imgs'):
        img = cv2.imread(p)
        if img is None: continue
        e = UWIE(img)
        crop, _ = morph_process(e)
        crop = cv2.resize(crop, IMG_SIZE)
        X.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)/255.0)
        y.append(label)
    X = np.array(X); y = np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.15, stratify=y)
    model = build_patternnet(input_shape=IMG_SIZE+(3,), n_classes=len(np.unique(y)))
    model.fit(X_train, y_train, validation_data=(X_val,y_val),
              epochs=25, batch_size=32)
    model.save('patternnet_model.h5')
    # save label encoder
    pickle.dump(le, open('label_encoder.pkl', 'wb'))

if __name__ == '__main__':
    paths, labels = load_image_paths(DATA_DIR)
    # choose approach:
    approach = 'cnn'  # 'bovw' or 'cnn'
    if approach == 'bovw':
        X, kmeans = build_bovw_for_dataset(paths)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder(); y = le.fit_transform(labels)
        train_bovw_classifier(X, y)
        pickle.dump({'kmeans':kmeans, 'le':le}, open('bovw_meta.pkl','wb'))
    else:
        train_cnn(paths, labels)
