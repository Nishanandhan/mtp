import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy import stats
from scipy.stats import skew
from scipy.stats import pearsonr
import statistics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

from flask import Flask, request, render_template
from flask.templating import render_template
from flask_sqlalchemy import SQLAlchemy

# Import for Migrations
from flask_migrate import Migrate, migrate

app = Flask(__name__)
app.debug = True
 
# adding configuration for using a sqlite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
 
# Creating an SQLAlchemy instance
db = SQLAlchemy(app)

# Models
class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    state = db.Column(db.String(100), unique=False, nullable=False)
    district = db.Column(db.String(100), unique=False, nullable=False)
    block = db.Column(db.String(100), unique=False, nullable=False)
    village = db.Column(db.String(100), unique=False, nullable=False)
    name = db.Column(db.String(100), unique=False, nullable=False)
    phone = db.Column(db.Integer, nullable=False)
    prev_crop = db.Column(db.String(100), unique=False, nullable=False)

    crop_group = db.Column(db.String(100), unique=False, nullable=True)
    crop = db.Column(db.String(100), unique=False, nullable=True)
    variety = db.Column(db.String(100), unique=False, nullable=True)
    season = db.Column(db.String(100), unique=False, nullable=True)
    soil_type = db.Column(db.String(100), unique=False, nullable=True)
    crop_duration = db.Column(db.String(100), unique=False, nullable=True)
    irrigation = db.Column(db.String(100), unique=False, nullable=True)
 
    # repr method represents how one object of this datatable
    # will look like
    def __repr__(self):
        return f"Name : {self.name}"

migrate = Migrate(app, db)

@app.route('/', methods =["GET", "POST"])
def index():    
    if request.method == "POST":
        state = request.form.get("state")
        district = request.form.get("district")
        block = request.form.get("block")
        village = request.form.get("village")
        name = request.form.get("name")
        phone = request.form.get("phone")
        prev_crop = request.form.get("prev_crop")
        print(state, district, block, village, name, phone, prev_crop)
        p = Profile(state=state, district=district, block=block, village=village, name=name, phone=phone, prev_crop=prev_crop)
        db.session.add(p)
        db.session.commit()
        return render_template("index2.html", id=p.id)
    else:
        return render_template("index.html")
        
@app.route('/predict_fertilizer', methods =["GET", "POST"])        
def index2():
    if request.method == "POST":
        pid = request.form.get("id")
        data = Profile.query.get(pid)

        data.crop_group = request.form.get("crop_group")
        data.crop = request.form.get("crop")
        data.variety = request.form.get("variety")
        data.season = request.form.get("season")
        data.soil_type = request.form.get("soil_type")
        data.crop_duration = request.form.get("crop_duration")
        data.irrigation = request.form.get("irrigation")
        
        db.session.commit()
        return render_template("index3.html", data=",".join([str(data.crop), str(data.variety)]))
    else:
        return render_template("index3.html")
        
# @app.route('/<crop_group>', methods =["GET", "POST"])        
# def index3(crop_group):
#     if request.method == "POST":
#         #fname = request.form.get("fname")
#         #lname = request.form.get("lname")       
#         #name = fname + " " + lname  
#         crop_group = request.form.get("crop_group")
#         return render_template("index2.html", name=crop_group)

@app.route('/predict', methods = ["GET", "POST"])
def predict():
    
    xrf_fe = float(request.form.get("xrf_fe"))
    xrf_k = float(request.form.get("xrf_k"))
    xrf_ti = float(request.form.get("xrf_ti"))
    xrf_ca = float(request.form.get("xrf_ca"))
    xrf_ba = float(request.form.get("xrf_ba"))
    xrf_zr = float(request.form.get("xrf_zr"))
    xrf_mn= float(request.form.get("xrf_mn"))
    xrf_rb = float(request.form.get("xrf_rb"))
    xrf_cr = float(request.form.get("xrf_cr"))
    xrf_sn = float(request.form.get("xrf_sn"))
    xrf_v = float(request.form.get("xrf_v"))
    xrf_ni = float(request.form.get("xrf_ni"))
    xrf_sr = float(request.form.get("xrf_sr"))
    xrf_zn = float(request.form.get("xrf_zn"))
    xrf_sb = float(request.form.get("xrf_sb"))
    xrf_cu = float(request.form.get("xrf_cu"))
    xrf_pb = float(request.form.get("xrf_pb"))
    xrf_ag = float(request.form.get("xrf_ag"))
    xrf_ga = float(request.form.get("xrf_ga"))

    ph = float(request.form.get("ph"))
    loi = float(request.form.get("loi"))
    oc = float(request.form.get("oc"))
    ec = float(request.form.get("ec"))

    agro_climatic_region = request.form.get("agro_climatic_region")
    parent_material = request.form.get("parent_material")
    order = request.form.get("order")
    productivity_potential = request.form.get("productivity_potential")

    full_inp = np.array([xrf_fe, xrf_k, xrf_ti, xrf_ca, xrf_ba, xrf_zr, xrf_mn, xrf_rb, xrf_cr, xrf_sn, xrf_v, xrf_ni, xrf_sr, xrf_zn, xrf_sb, xrf_cu, xrf_pb, xrf_ag, xrf_ga, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ph, loi, oc, ec ])

    if agro_climatic_region == "Coastal saline":
        full_inp[19]=1
    elif agro_climatic_region == "Gangetic alluvial":
        full_inp[20]=1
    elif agro_climatic_region == "Red lateritic":
        full_inp[21]=1
    elif agro_climatic_region == "Terai-Teesta alluvial":
        full_inp[22]=1
    elif agro_climatic_region == "Vindhyachal alluvial":
        full_inp[23]=1

    if parent_material == "Granite gneiss":
        full_inp[24]=1
    elif parent_material == "Deltaic alluvium":
        full_inp[25]=1
    elif parent_material == "Old alluvium":
        full_inp[26]=1

    if order == "Alfisol":
        full_inp[27]=1
    elif order == "Entisol":
        full_inp[28]=1

    if productivity_potential == "High":
        full_inp[29]=1
    elif productivity_potential == "Low":
        full_inp[30]=1
    
    #print(full_inp)
    #print(agro_climatic_region,parent_material,order,productivity_potential)

    n_result = func_n(full_inp)
    k_result = func_k(full_inp)
    ca_result = func_ca(full_inp)
    mg_result = func_mg(full_inp)
    zn_result = func_zn(full_inp)
    cu_result = func_cu(full_inp)
    fe_result = func_fe(full_inp)
    mn_result = func_mn(full_inp)
    b_result = func_b(full_inp)

    tmp = str(request.form.get("crop_variety")).split(",")
    print(str(request.form.get("crop_variety")), tmp)
    crop = tmp[0]
    variety=tmp[1]

    result = {"result": [n_result, k_result, ca_result, mg_result, zn_result, cu_result, fe_result, mn_result, b_result]}

    return render_template("result.html", data=result, crop=crop, variety=variety)


    #######

    # with open('static/models/petiole_potassium_model.pkl', 'rb') as f:
    #     k_model = pickle.load(f)
    # with open('static/models/petiole_potassium_stat.txt', 'rb') as f:
    #     k_stat = pickle.load(f)
    #     k_mean = k_stat["mean"]
    #     k_stddev = k_stat["std_dev"]

    # print(k_mean, k_stddev, k_model)

    # inp = np.array([xrf_k, xrf_ca, xrf_mn])
    # for i in range(3):
    #     inp[i] = (inp[i]-float(k_mean[i]))/float(k_stddev[i])
    # inp.resize(1,3)

    # result = k_model.predict(inp)
    # print(result)

    # #######

    # return render_template("result.html", result=result)


def func_n(full_inp):
    with open('static/models/n_rf.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('static/models/n_svr.pkl', 'rb') as f:
        svr = pickle.load(f)
    with open('static/models/n_mlr.pkl', 'rb') as f:
        mlr = pickle.load(f)

    #                                                                                                                                                       19 20 21 22 23 24 25 26 27 28 29 30
    #  0       1       2        3       4       5       6       7       8       9       10      11     12      13      14   15          16      17      18  0  1  2  3  4  6  7  8  10 11 13 14 31  32   33  34  
    # xrf_fe, xrf_k, xrf_ti, xrf_ca, xrf_ba, xrf_zr, xrf_mn, xrf_rb, xrf_cr, xrf_sn, xrf_v, xrf_ni, xrf_sr, xrf_zn, xrf_sb, xrf_cu, xrf_pb, xrf_ag, xrf_ga, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ph, loi, oc, ec     
    #input_mlr = np.array([full_inp[0], full_inp[3], full_inp[5], full_inp[9], full_inp[10], full_inp[11], full_inp[15], full_inp[16], full_inp[20], full_inp[21], full_inp[22], full_inp[23], full_inp[25], full_inp[29], full_inp[30], full_inp[31], full_inp[32]])
    select_arr = np.array([ True, False, False,  True, False,  True, False, False, False,
                            True,  True,  True, False, False, False,  True,  True, False,
                           False, False,  True,  True,  True,  True, False,  True, False,
                           False, False,  True,  True,  True,  True, False, False])
    input_mlr = np.array([])
    for i in range(0,select_arr.size):
        if select_arr[i]:
            input_mlr = np.append(input_mlr,full_inp[i])

    prd1 = rf.predict([full_inp])
    prd2 = svr.predict([full_inp])
    prd3 = mlr.predict([input_mlr])
    pred_value = (prd1+prd2+prd3)/3

    if pred_value<280:
        pred_range = "Low"
    elif pred_value>=280 and pred_value<=560:
        pred_range = "Medium"
    elif pred_value>560:
        pred_range = "High"

    result = [round(float(pred_value[0]),2), pred_range]
    #print(result)
    #print(input_mlr)
    return result



def func_k(full_inp):
    with open('static/models/k_rf.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('static/models/k_svr.pkl', 'rb') as f:
        svr = pickle.load(f)
    with open('static/models/k_mlr.pkl', 'rb') as f:
        mlr = pickle.load(f)

    #                                                                                                                                                       19 20 21 22 23 24 25 26 27 28 29 30
    #  0       1       2        3       4       5       6       7       8       9       10      11     12      13      14   15          16      17      18  0  1  2  3  4  6  7  8  10 11 13 14 31  32   33  34  
    # xrf_fe, xrf_k, xrf_ti, xrf_ca, xrf_ba, xrf_zr, xrf_mn, xrf_rb, xrf_cr, xrf_sn, xrf_v, xrf_ni, xrf_sr, xrf_zn, xrf_sb, xrf_cu, xrf_pb, xrf_ag, xrf_ga, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ph, loi, oc, ec     
    #input_mlr = np.array([full_inp[0], full_inp[3], full_inp[5], full_inp[9], full_inp[10], full_inp[11], full_inp[15], full_inp[16], full_inp[20], full_inp[21], full_inp[22], full_inp[23], full_inp[25], full_inp[29], full_inp[30], full_inp[31], full_inp[32]])
    select_arr = np.array([ True, False, False,  True, False,  True, False,  True, False,
                            True,  True, False, False, False, False, False,  True, False,
                            True, False,  True,  True,  True,  True, False,  True,  True,
                           False, False,  True, False,  True, False, False,  True])
    input_mlr = np.array([])
    for i in range(0,select_arr.size):
        if select_arr[i]:
            input_mlr = np.append(input_mlr,full_inp[i])

    prd1 = rf.predict([full_inp])
    prd2 = svr.predict([full_inp])
    prd3 = mlr.predict([input_mlr])
    pred_value = (prd1+prd2+prd3)/3

    if pred_value<151:
        pred_range = "Low"
    elif pred_value>=151 and pred_value<=250:
        pred_range = "Medium"
    elif pred_value>250:
        pred_range = "High"

    result = [round(float(pred_value[0]),2), pred_range]
    #print(result)
    #print(input_mlr)
    return result



def func_ca(full_inp):
    with open('static/models/ca_rf.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('static/models/ca_svr.pkl', 'rb') as f:
        svr = pickle.load(f)
    with open('static/models/ca_mlr.pkl', 'rb') as f:
        mlr = pickle.load(f)

    #                                                                                                                                                       19 20 21 22 23 24 25 26 27 28 29 30
    #  0       1       2        3       4       5       6       7       8       9       10      11     12      13      14   15          16      17      18  0  1  2  3  4  6  7  8  10 11 13 14 31  32   33  34  
    # xrf_fe, xrf_k, xrf_ti, xrf_ca, xrf_ba, xrf_zr, xrf_mn, xrf_rb, xrf_cr, xrf_sn, xrf_v, xrf_ni, xrf_sr, xrf_zn, xrf_sb, xrf_cu, xrf_pb, xrf_ag, xrf_ga, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ph, loi, oc, ec     
    #input_mlr = np.array([full_inp[0], full_inp[3], full_inp[5], full_inp[9], full_inp[10], full_inp[11], full_inp[15], full_inp[16], full_inp[20], full_inp[21], full_inp[22], full_inp[23], full_inp[25], full_inp[29], full_inp[30], full_inp[31], full_inp[32]])
    select_arr = np.array([ True,  True, False,  True, False, False,  True, False,  True,
                            True, False,  True,  True,  True,  True,  True,  True, False,
                           False, False,  True, False,  True,  True, False, False, False,
                           False, False,  True, False,  True, False, False, False])
    input_mlr = np.array([])
    for i in range(0,select_arr.size):
        if select_arr[i]:
            input_mlr = np.append(input_mlr,full_inp[i])

    prd1 = rf.predict([full_inp])
    prd2 = svr.predict([full_inp])
    prd3 = mlr.predict([input_mlr])
    pred_value = (prd1+prd2+prd3)/3

    if pred_value<2000:
        pred_range = "Low"
    elif pred_value>=2000 and pred_value<=4000:
        pred_range = "Medium"
    elif pred_value>4000:
        pred_range = "High"

    result = [round(float(pred_value[0]),2), pred_range]
    #print(result)
    #print(input_mlr)
    return result



def func_mg(full_inp):
    with open('static/models/mg_rf.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('static/models/mg_svr.pkl', 'rb') as f:
        svr = pickle.load(f)
    with open('static/models/mg_mlr.pkl', 'rb') as f:
        mlr = pickle.load(f)

    #                                                                                                                                                       19 20 21 22 23 24 25 26 27 28 29 30
    #  0       1       2        3       4       5       6       7       8       9       10      11     12      13      14   15          16      17      18  0  1  2  3  4  6  7  8  10 11 13 14 31  32   33  34  
    # xrf_fe, xrf_k, xrf_ti, xrf_ca, xrf_ba, xrf_zr, xrf_mn, xrf_rb, xrf_cr, xrf_sn, xrf_v, xrf_ni, xrf_sr, xrf_zn, xrf_sb, xrf_cu, xrf_pb, xrf_ag, xrf_ga, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ph, loi, oc, ec     
    #input_mlr = np.array([full_inp[0], full_inp[3], full_inp[5], full_inp[9], full_inp[10], full_inp[11], full_inp[15], full_inp[16], full_inp[20], full_inp[21], full_inp[22], full_inp[23], full_inp[25], full_inp[29], full_inp[30], full_inp[31], full_inp[32]])
    select_arr = np.array([False, False, False,  True, False, False,  True,  True,  True,
                           False, False, False,  True,  True, False,  True,  True, False,
                           False,  True, False, False,  True, False, False, False,  True,
                            True,  True,  True,  True, False, False,  True,  True])
    input_mlr = np.array([])
    for i in range(0,select_arr.size):
        if select_arr[i]:
            input_mlr = np.append(input_mlr,full_inp[i])

    prd1 = rf.predict([full_inp])
    prd2 = svr.predict([full_inp])
    prd3 = mlr.predict([input_mlr])
    pred_value = (prd1+prd2+prd3)/3

    if pred_value<396:
        pred_range = "Low"
    elif pred_value>=396 and pred_value<=996:
        pred_range = "Medium"
    elif pred_value>996:
        pred_range = "High"

    result = [round(float(pred_value[0]),2), pred_range]
    #print(result)
    #print(input_mlr)
    return result



def func_zn(full_inp):
    with open('static/models/zn_rf.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('static/models/zn_svr.pkl', 'rb') as f:
        svr = pickle.load(f)
    with open('static/models/zn_mlr.pkl', 'rb') as f:
        mlr = pickle.load(f)

    #                                                                                                                                                       19 20 21 22 23 24 25 26 27 28 29 30
    #  0       1       2        3       4       5       6       7       8       9       10      11     12      13      14   15          16      17      18  0  1  2  3  4  6  7  8  10 11 13 14 31  32   33  34  
    # xrf_fe, xrf_k, xrf_ti, xrf_ca, xrf_ba, xrf_zr, xrf_mn, xrf_rb, xrf_cr, xrf_sn, xrf_v, xrf_ni, xrf_sr, xrf_zn, xrf_sb, xrf_cu, xrf_pb, xrf_ag, xrf_ga, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ph, loi, oc, ec     
    #input_mlr = np.array([full_inp[0], full_inp[3], full_inp[5], full_inp[9], full_inp[10], full_inp[11], full_inp[15], full_inp[16], full_inp[20], full_inp[21], full_inp[22], full_inp[23], full_inp[25], full_inp[29], full_inp[30], full_inp[31], full_inp[32]])
    select_arr = np.array([ True, False,  True,  True, False,  True,  True,  True,  True,
                           False, False, False, False,  True, False, False,  True, False,
                           False,  True,  True, False,  True,  True, False,  True,  True,
                           False, False,  True, False, False,  True, False, False])
    input_mlr = np.array([])
    for i in range(0,select_arr.size):
        if select_arr[i]:
            input_mlr = np.append(input_mlr,full_inp[i])

    prd1 = rf.predict([full_inp])
    prd2 = svr.predict([full_inp])
    prd3 = mlr.predict([input_mlr])
    pred_value = (prd1+prd2+prd3)/3

    if pred_value<1.21:
        pred_range = "Low"
    elif pred_value>=1.21 and pred_value<=2.4:
        pred_range = "Medium"
    elif pred_value>2.4:
        pred_range = "High"

    result = [round(float(pred_value[0]),2), pred_range]
    #print(result)
    #print(input_mlr)
    return result



def func_cu(full_inp):
    with open('static/models/cu_rf.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('static/models/cu_svr.pkl', 'rb') as f:
        svr = pickle.load(f)
    with open('static/models/cu_mlr.pkl', 'rb') as f:
        mlr = pickle.load(f)

    #                                                                                                                                                       19 20 21 22 23 24 25 26 27 28 29 30
    #  0       1       2        3       4       5       6       7       8       9       10      11     12      13      14   15          16      17      18  0  1  2  3  4  6  7  8  10 11 13 14 31  32   33  34  
    # xrf_fe, xrf_k, xrf_ti, xrf_ca, xrf_ba, xrf_zr, xrf_mn, xrf_rb, xrf_cr, xrf_sn, xrf_v, xrf_ni, xrf_sr, xrf_zn, xrf_sb, xrf_cu, xrf_pb, xrf_ag, xrf_ga, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ph, loi, oc, ec     
    #input_mlr = np.array([full_inp[0], full_inp[3], full_inp[5], full_inp[9], full_inp[10], full_inp[11], full_inp[15], full_inp[16], full_inp[20], full_inp[21], full_inp[22], full_inp[23], full_inp[25], full_inp[29], full_inp[30], full_inp[31], full_inp[32]])
    select_arr = np.array([False, False, False,  True, False,  True,  True, False, False,
                            True, False, False,  True,  True, False,  True,  True, False,
                            True])
    input_mlr = np.array([])
    for i in range(0,select_arr.size):
        if select_arr[i]:
            input_mlr = np.append(input_mlr,full_inp[i])

    prd1 = rf.predict([full_inp[0:19]])
    prd2 = svr.predict([full_inp[0:19]])
    prd3 = mlr.predict([input_mlr])
    pred_value = (prd1+prd2+prd3)/3

    if pred_value<0.41:
        pred_range = "Low"
    elif pred_value>=0.41 and pred_value<=1.2:
        pred_range = "Medium"
    elif pred_value>1.2:
        pred_range = "High"

    result = [round(float(pred_value[0]),2), pred_range]
    #print(result)
    #print(input_mlr)
    return result



def func_fe(full_inp):
    with open('static/models/fe_rf.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('static/models/fe_svr.pkl', 'rb') as f:
        svr = pickle.load(f)
    with open('static/models/fe_mlr.pkl', 'rb') as f:
        mlr = pickle.load(f)

    #                                                                                                                                                       19 20 21 22 23 24 25 26 27 28 29 30
    #  0       1       2        3       4       5       6       7       8       9       10      11     12      13      14   15          16      17      18  0  1  2  3  4  6  7  8  10 11 13 14 31  32   33  34  
    # xrf_fe, xrf_k, xrf_ti, xrf_ca, xrf_ba, xrf_zr, xrf_mn, xrf_rb, xrf_cr, xrf_sn, xrf_v, xrf_ni, xrf_sr, xrf_zn, xrf_sb, xrf_cu, xrf_pb, xrf_ag, xrf_ga, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ph, loi, oc, ec     
    #input_mlr = np.array([full_inp[0], full_inp[3], full_inp[5], full_inp[9], full_inp[10], full_inp[11], full_inp[15], full_inp[16], full_inp[20], full_inp[21], full_inp[22], full_inp[23], full_inp[25], full_inp[29], full_inp[30], full_inp[31], full_inp[32]])
    select_arr = np.array([ True, False, False,  True, False, False, False, False,  True,
                            True, False,  True,  True, False, False, False,  True, False,
                           False,  True,  True, False,  True,  True,  True, False,  True,
                           False,  True, False,  True,  True, False, False,  True])
    input_mlr = np.array([])
    for i in range(0,select_arr.size):
        if select_arr[i]:
            input_mlr = np.append(input_mlr,full_inp[i])

    prd1 = rf.predict([full_inp])
    prd2 = svr.predict([full_inp])
    prd3 = mlr.predict([input_mlr])
    pred_value = (prd1+prd2+prd3)/3

    if pred_value<9.1:
        pred_range = "Low"
    elif pred_value>=9.1 and pred_value<=27:
        pred_range = "Medium"
    elif pred_value>27:
        pred_range = "High"

    result = [round(float(pred_value[0]),2), pred_range]
    #print(result)
    #print(input_mlr)
    return result



def func_mn(full_inp):
    with open('static/models/mn_rf.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('static/models/mn_svr.pkl', 'rb') as f:
        svr = pickle.load(f)
    with open('static/models/mn_mlr.pkl', 'rb') as f:
        mlr = pickle.load(f)

    #                                                                                                                                                       19 20 21 22 23 24 25 26 27 28 29 30
    #  0       1       2        3       4       5       6       7       8       9       10      11     12      13      14   15          16      17      18  0  1  2  3  4  6  7  8  10 11 13 14 31  32   33  34  
    # xrf_fe, xrf_k, xrf_ti, xrf_ca, xrf_ba, xrf_zr, xrf_mn, xrf_rb, xrf_cr, xrf_sn, xrf_v, xrf_ni, xrf_sr, xrf_zn, xrf_sb, xrf_cu, xrf_pb, xrf_ag, xrf_ga, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ph, loi, oc, ec     
    #input_mlr = np.array([full_inp[0], full_inp[3], full_inp[5], full_inp[9], full_inp[10], full_inp[11], full_inp[15], full_inp[16], full_inp[20], full_inp[21], full_inp[22], full_inp[23], full_inp[25], full_inp[29], full_inp[30], full_inp[31], full_inp[32]])
    select_arr = np.array([False,  True, False,  True, False,  True,  True, False,  True,
                           False, False, False, False,  True,  True,  True,  True, False,
                           False, False, False,  True,  True,  True, False,  True,  True,
                           False, False,  True, False,  True, False,  True, False])
    input_mlr = np.array([])
    for i in range(0,select_arr.size):
        if select_arr[i]:
            input_mlr = np.append(input_mlr,full_inp[i])

    prd1 = rf.predict([full_inp])
    prd2 = svr.predict([full_inp])
    prd3 = mlr.predict([input_mlr])
    pred_value = (prd1+prd2+prd3)/3

    if pred_value<4.1:
        pred_range = "Low"
    elif pred_value>=4.1 and pred_value<=16:
        pred_range = "Medium"
    elif pred_value>16:
        pred_range = "High"

    result = [round(float(pred_value[0]),2), pred_range]
    #print(result)
    #print(input_mlr)
    return result



def func_b(full_inp):
    with open('static/models/b_rf.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('static/models/b_svr.pkl', 'rb') as f:
        svr = pickle.load(f)
    with open('static/models/b_mlr.pkl', 'rb') as f:
        mlr = pickle.load(f)

    #                                                                                                                                                       19 20 21 22 23 24 25 26 27 28 29 30
    #  0       1       2        3       4       5       6       7       8       9       10      11     12      13      14   15          16      17      18  0  1  2  3  4  6  7  8  10 11 13 14 31  32   33  34  
    # xrf_fe, xrf_k, xrf_ti, xrf_ca, xrf_ba, xrf_zr, xrf_mn, xrf_rb, xrf_cr, xrf_sn, xrf_v, xrf_ni, xrf_sr, xrf_zn, xrf_sb, xrf_cu, xrf_pb, xrf_ag, xrf_ga, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ph, loi, oc, ec     
    #input_mlr = np.array([full_inp[0], full_inp[3], full_inp[5], full_inp[9], full_inp[10], full_inp[11], full_inp[15], full_inp[16], full_inp[20], full_inp[21], full_inp[22], full_inp[23], full_inp[25], full_inp[29], full_inp[30], full_inp[31], full_inp[32]])
    select_arr = np.array([False,  True,  True, False, False,  True, False,  True,  True,
                            True, False,  True,  True, False, False, False,  True, False,
                           False])
    input_mlr = np.array([])
    for i in range(0,select_arr.size):
        if select_arr[i]:
            input_mlr = np.append(input_mlr,full_inp[i])

    prd1 = rf.predict([full_inp[0:19]])
    prd2 = svr.predict([full_inp[0:19]])
    prd3 = mlr.predict([input_mlr])
    pred_value = (prd1+prd2+prd3)/3

    if pred_value<1:
        pred_range = "Low"
    elif pred_value>=1 and pred_value<=2:
        pred_range = "Medium"
    elif pred_value>2:
        pred_range = "High"

    result = [round(float(pred_value[0]),2), pred_range]
    #print(result)
    #print(input_mlr)
    return result



@app.route('/database', methods =["GET", "POST"]) 
def database():    
    profiles = Profile.query.all()
    return render_template('database.html', profiles=profiles)
        
if __name__=='__main__':
    app.run(debug=True)