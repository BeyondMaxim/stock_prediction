from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, mean_squared_error
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from math import sqrt

###
def train_and_backtest_gbc(X,Y,P,RET,current_month_id,training_window,params):
    
    max_leaf_nodes   = params['max_leaf_nodes']
    max_depth        = params['max_depth']
    min_samples_leaf = params['min_samples_leaf']
    n_estimators     = params['n_estimators']
    learning_rate    = params['learning_rate']
    max_features     = params["max_features"]
    subsample        = params["subsample"] 

    D = P.unique()
    
    X_train =  X[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    Y_train =  Y[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    
    X_test =   X[P==D[current_month_id]]
    R_test = RET[P==D[current_month_id]]
    
    gbc = GradientBoostingClassifier(max_leaf_nodes = max_leaf_nodes,
                                     max_depth = max_depth,
                                     min_samples_leaf = min_samples_leaf, 
                                     n_estimators = n_estimators,
                                     learning_rate = learning_rate,
                                     max_features = max_features,
                                     subsample = subsample)
    gbc.fit(X_train,Y_train)
    
    Y_insample = gbc.predict(X_train)
    Y_proba    = gbc.predict_proba(X_test)[:,1]
    
    insample_mse = 100*np.mean((Y_insample==Y_train)**2)
    
    ret_formward_1m =  np.mean(R_test[Y_proba>np.percentile(Y_proba,80)])-np.mean(R_test[Y_proba<=np.percentile(Y_proba,20)])
    
    out = pd.DataFrame(columns=['date','insample mse','forward return 1m'])
                           
    out.loc[0,'date']              = D[current_month_id]
    out.loc[0,'insample mse']      = insample_mse
    out.loc[0,'forward return 1m'] = ret_formward_1m
    
    return out
###

###
def ptfo_metrics(backtest):
    monthly_return = backtest.pct_change(1)
    monthly_return=monthly_return.dropna()
    avg_monthly_return = monthly_return.mean()
    period_return =((monthly_return+1).cumprod()-1)[-1]
    yearly_return= ((1+period_return)**(12/len(backtest.index)))-1
    volatility = monthly_return.std() * np.sqrt(12) * 100
    sharpe = (yearly_return*100)/volatility
    return print(
        'average monthly return:', round(avg_monthly_return*100,2), 
        '\n',
        'cumulative return:', round(period_return*100,2),
        '\n',
        'yearly return:', round(yearly_return*100,2),
        '\n',
        'volatility:', round(volatility,2),
        '\n',
        'sharpe:', round(sharpe,2)
    )
###

def ptfo_metrics_v2(backtest):
    
    monthly_returns = backtest.pct_change(1)
    monthly_returns = monthly_returns.dropna()
    cumulative_return =((monthly_returns+1).cumprod()-1)
    annualized_return = ((1+cumulative_return[-1])**(12/len(backtest)))-1
    
    cum_returns = np.array(backtest) / backtest[0]
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (peak - cum_returns) / peak
    mdd = np.max(drawdown)
    
    colmar = annualized_return/mdd
    volatility = monthly_returns.std() * np.sqrt(12) * 100
    sharpe = (annualized_return*100)/volatility
    return print(
        ' cumulative return:', round(cumulative_return[-1]*100,2),
        '\n',
        'annualized return:', round(annualized_return*100,2),
        '\n',
        'volatility:', round(volatility,2),
        '\n',
        'mdd:', round(mdd*100,2),
        '\n', 
        'sharpe:', round(sharpe,2),
        '\n', 
        'colmar:', round(colmar,2)
    )

###
def calculate_ic(Y_proba, R_test):
    """
    Calculate the information coefficient (IC) between predicted probabilities and actual returns.
    
    Parameters:
    -----------
    Y_proba : array-like, shape (n_samples,)
        Predicted probabilities of the positive class (class 1).
    R_test : array-like, shape (n_samples,)
        Actual returns corresponding to the same observations as Y_proba.
    
    Returns:
    --------
    ic : float
        The information coefficient between Y_proba and R_test.
    """
    # Calculate the Pearson correlation coefficient between Y_proba and R_test
    corr = np.corrcoef(Y_proba, R_test)[0, 1]
    
    # Calculate the number of observations in Y_proba
    n = len(Y_proba)
    
    # Calculate the information coefficient
    ic = corr * np.sqrt(n)
    
    return ic

###

def train_and_backtest_xgb(X,Y,P,RET,current_month_id,training_window,params):
    
    n_estimators     = params['n_estimators']

    D = P.unique()
    
    X_train =  X[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    Y_train =  Y[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    
    X_test =   X[P==D[current_month_id]]
    R_test = RET[P==D[current_month_id]]
    
    xgb_m = xgb.XGBClassifier( 
                            n_estimators = n_estimators,
                            )
    
    xgb_m.fit(X_train,Y_train)
    
    Y_insample = gbc.predict(X_train)
    Y_proba    = gbc.predict_proba(X_test)[:,1]
    
    insample_mse = 100*np.mean((Y_insample==Y_train)**2)
    
    ret_formward_1m =  np.mean(R_test[Y_proba>np.percentile(Y_proba,80)])-np.mean(R_test[Y_proba<=np.percentile(Y_proba,20)])
    
    out = pd.DataFrame(columns=['date','insample mse','forward return 1m'])
                           
    out.loc[0,'date']              = D[current_month_id]
    out.loc[0,'insample mse']      = insample_mse
    out.loc[0,'forward return 1m'] = ret_formward_1m
    
    return out
###

def train_and_backtest_model(X,Y,P,RET,current_month_id,training_window, params, model_name):
    
    D = P.unique()
    
    X_train =  X[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    Y_train =  Y[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    
    X_test =   X[P==D[current_month_id]]
    R_test = RET[P==D[current_month_id]]
    
    model = model_name(**params,random_state=42) 
    
    fit = model.fit(X_train, Y_train)
    Y_proba    = fit.predict_proba(X_test)[:,1]
    if round(Y_proba.mean(), 4) == round(Y_proba[0], 4):
      Y_proba = add_jitter(Y_proba)
    
    Y_insample = fit.predict(X_train)
    insample_mse = mean_squared_error(Y_train, Y_insample)
    insample_rmse = sqrt(insample_mse)
    
    Y_outsample = fit.predict(X_test)
    outsample_mse = mean_squared_error(R_test, Y_outsample)
    outsample_rmse = sqrt(outsample_mse)
                        
    acc = accuracy_score(Y_train, Y_insample)
    f1 = f1_score(Y_train, Y_insample)
    
    ret_formward_1m =  np.mean(R_test[Y_proba>np.percentile(Y_proba,80)])-np.mean(R_test[Y_proba<=np.percentile(Y_proba,20)])
    
    ic = calculate_ic(Y_proba, R_test)
    
    out = pd.DataFrame(columns=['date','insample rmse','outsample rmse','accuracy','f1-score','forward return 1m', 'IC'])
                           
    out.loc[0,'date']              = D[current_month_id]
    out.loc[0,'insample rmse']     = insample_rmse
    out.loc[0,'outsample rmse']    = outsample_rmse
    out.loc[0,'accuracy']          = acc
    out.loc[0,'f1-score']          = f1
    out.loc[0,'forward return 1m'] = ret_formward_1m
    out.loc[0,'IC']                = ic
    
    return out

###

















def backtest_top_vs_bottom_quintile(scores,forward_returns,periods):
    
    #rank stocks in quintiles within each period and assign 1 to top 20% and -1 to bottom 20%
    top_vs_bottom_label = scores.groupby(periods).transform(lambda x: 1*(x>x.quantile(0.8))-1*(x<=x.quantile(0.2)))
    
    #take the average return of the top and bottom portfios and subtract to calculate top vs bottom return
    top_vs_bottom_return = forward_returns.groupby([periods,top_vs_bottom_label]).mean().unstack(level=1).apply(lambda x: x[1.0]-x[-1.0],axis=1)

    top_vs_bottom_return.index = pd.to_datetime(top_vs_bottom_return.index.astype('int'), format='%Y%m%d')
    
    #create index from return series
    top_vs_bottom_index = 100*(1+top_vs_bottom_return.shift(1)/100).cumprod()
    
    top_vs_bottom_index.iloc[0] = 100
    
    return top_vs_bottom_index


def train_and_backtest(X,Y,P,RET,current_month_id,training_window,params):
    
    max_leaf_nodes   = params['max_leaf_nodes']
    min_samples_leaf = params['min_samples_leaf']
    n_estimators     = params['n_estimators']
    learning_rate    = params['learning_rate']
    max_features     = params['max_features']
    subsample        = params['subsample']

    D = P.unique()
    
    X_train =  X[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    Y_train =  Y[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    
    X_test =   X[P==D[current_month_id]]
    R_test = RET[P==D[current_month_id]]
    
    gbc = GradientBoostingClassifier(max_leaf_nodes = max_leaf_nodes, min_samples_leaf = min_samples_leaf, n_estimators = n_estimators,learning_rate = learning_rate, max_features = max_features, subsample = subsample)
    gbc.fit(X_train,Y_train)
    
    Y_insample = gbc.predict(X_train)
    Y_proba    = gbc.predict_proba(X_test)[:,1]
    
    insample_mse = 100*np.mean((Y_insample==Y_train)**2)
    
    ret_formward_1m =  np.mean(R_test[Y_proba>np.percentile(Y_proba,80)])-np.mean(R_test[Y_proba<=np.percentile(Y_proba,20)])
    
    out = pd.DataFrame(columns=['date','insample mse','forward return 1m'])
                           
    out.loc[0,'date']              = D[current_month_id]
    out.loc[0,'insample mse']      = insample_mse
    out.loc[0,'forward return 1m'] = ret_formward_1m
    
    return out



def train_and_backtest_RF_man(X,Y,P,RET,current_month_id,training_window,params):
    
    max_features     = params['max_features']
    bootstrap        = params['bootstrap']
    random_state     = params['random_state']

    D = P.unique()
    
    X_train =  X[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    Y_train =  Y[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    
    X_test =   X[P==D[current_month_id]]
    R_test = RET[P==D[current_month_id]]
    
    rf_man_model = RandomForestClassifier(max_depth=3, n_estimators=1, max_features=max_features, bootstrap=bootstrap, n_jobs=-1, random_state=random_state)
    rf_man = rf_man_model.fit(X_train, Y_train)
    
    Y_insample = rf_man.predict(X_train)
    Y_proba    = rf_man.predict_proba(X_test)[:,1]
    
    insample_mse = 100*np.mean((Y_insample==Y_train)**2)
    
    ret_formward_1m =  np.mean(R_test[Y_proba>np.percentile(Y_proba,80)])-np.mean(R_test[Y_proba<=np.percentile(Y_proba,20)])
    
    out = pd.DataFrame(columns=['date','insample mse','forward return 1m'])
                           
    out.loc[0,'date']              = D[current_month_id]
    out.loc[0,'insample mse']      = insample_mse
    out.loc[0,'forward return 1m'] = ret_formward_1m
    
    return out


def train_and_backtest_data_leakage(X,Y,P,RET,current_month_id,training_window,params):
    
    max_leaf_nodes   = params['max_leaf_nodes']
    min_samples_leaf = params['min_samples_leaf']
    n_estimators     = params['n_estimators']
    learning_rate    = params['learning_rate']
    max_features     = params['max_features']
    subsample        = params['subsample']

    D = P.unique()
    
    X_train =  X[(P>=D[current_month_id-training_window]) & (P<=D[current_month_id]) & (Y!=0)].copy()
    Y_train =  Y[(P>=D[current_month_id-training_window]) & (P<=D[current_month_id]) & (Y!=0)].copy()
    
    X_test =   X[P==D[current_month_id]]
    R_test = RET[P==D[current_month_id]]
    
    gbc = GradientBoostingClassifier(max_leaf_nodes = max_leaf_nodes, min_samples_leaf = min_samples_leaf, n_estimators = n_estimators,
                                     learning_rate = learning_rate, max_features = max_features, subsample = subsample)
    
    gbc.fit(X_train,Y_train)
    
    Y_insample = gbc.predict(X_train)
    Y_proba    = gbc.predict_proba(X_test)[:,1]
    
    insample_mse = 100*np.mean((Y_insample==Y_train)**2)
    
    ret_formward_1m =  np.mean(R_test[Y_proba>np.percentile(Y_proba,80)])-np.mean(R_test[Y_proba<=np.percentile(Y_proba,20)])
    
    out = pd.DataFrame(columns=['date','insample mse','forward return 1m'])
                           
    out.loc[0,'date']              = D[current_month_id]
    out.loc[0,'insample mse']      = insample_mse
    out.loc[0,'forward return 1m'] = ret_formward_1m
     
    return out


def train_and_backtest_regr(X,Y,P,RET,current_month_id,training_window):

    D = P.unique()
    
    X_train =  X[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    Y_train =  Y[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    X_test =   X[P==D[current_month_id]]
    R_test = RET[P==D[current_month_id]]
    
    regr = LinearRegression()
    regr.fit(X_train,Y_train)
    
    Y_insample = regr.predict(X_train)
    Y_outsample = regr.predict(X_test)
    
    insample_mse = 100*np.mean((Y_insample==Y_train)**2)
    ret_formward_1m =  np.mean(R_test[Y_outsample>np.percentile(Y_outsample,80)])-np.mean(R_test[Y_outsample<=np.percentile(Y_outsample,20)])
    out = pd.DataFrame(columns=['date','insample mse','forward return 1m'])
                           
    out.loc[0,'date']              = D[current_month_id]
    out.loc[0,'insample mse']      = insample_mse
    out.loc[0,'forward return 1m'] = ret_formward_1m
    
    return out



def train_and_backtest_RF_def(X,Y,P,RET,current_month_id,training_window, params):

    random_state = params['random_state']
    
    D = P.unique()
    
    X_train =  X[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    Y_train =  Y[(P>=D[current_month_id-training_window]) & (P<D[current_month_id]) & (Y!=0)].copy()
    
    X_test =   X[P==D[current_month_id]]
    R_test = RET[P==D[current_month_id]]
    
    rf_def_model = RandomForestClassifier(random_state=random_state)  # default params
    rf_def = rf_def_model.fit(X_train, Y_train)
    
    Y_insample = rf_def.predict(X_train)
    Y_proba    = rf_def.predict_proba(X_test)[:,1]
    
    insample_mse = 100*np.mean((Y_insample==Y_train)**2)
    
    ret_formward_1m =  np.mean(R_test[Y_proba>np.percentile(Y_proba,80)])-np.mean(R_test[Y_proba<=np.percentile(Y_proba,20)])
    
    out = pd.DataFrame(columns=['date','insample mse','forward return 1m'])
                           
    out.loc[0,'date']              = D[current_month_id]
    out.loc[0,'insample mse']      = insample_mse
    out.loc[0,'forward return 1m'] = ret_formward_1m
    
    return out



