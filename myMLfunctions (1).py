from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd
import numpy as np

def backtest_top_vs_bottom_quintile(scores,forward_returns,periods):
    
    #rank stocks in quintiles withineach period and assign 1 to top 20% and -1 to bottom 20%
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