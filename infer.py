# infer.py
import cv2, numpy as np, pickle
from uwie import UWIE
from morph import morph_process
from features import make_detector, extract_descriptors, compute_bovw_h
from tensorflow.keras.models import load_model

def predict_image_streamlit(img_bgr, approach='cnn'):
    # img_bgr: BGR image loaded by cv2
    e = UWIE(img_bgr)
    crop, mask = morph_process(e)
    if approach == 'cnn':
        model = load_model('patternnet_model.h5')
        import pickle
        le = pickle.load(open('label_encoder.pkl','rb'))
        x = cv2.resize(crop, (64,64))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255.0
        x = np.expand_dims(x,0)
        proba = model.predict(x)[0]
        idx = np.argmax(proba)
        label = le.inverse_transform([idx])[0]
        return label, proba[idx]
    else:
        meta = pickle.load(open('bovw_meta.pkl','rb'))
        kmeans = meta['kmeans']; le = meta['le']
        det = make_detector()
        des = extract_descriptors(crop, det)
        hist = compute_bovw_hist(des, kmeans).reshape(1,-1)
        bm = pickle.load(open('bovw_model.pkl','rb'))
        mask = bm['mask']; clf = bm['clf']
        hist_sel = hist[:, mask==1]
        pred = clf.predict(hist_sel)[0]
        label = le.inverse_transform([pred])[0]
        return label, None
