"""
A non-blending lightGBM model that incorporates portions and ideas from various public kernels.
"""
WHERE = 'aws'

if WHERE=='kaggle':
	inpath = '../input/'
	suffix = ''
	outpath = ''
	savepath = ''
elif WHERE=='gcloud':
	inpath = '../.kaggle/competitions/talkingdata-adtracking-fraud-detection/'
	suffix = '.zip'
	outpath = '../sub/'
	savepath = '../data/'
else:
	inpath = './'
	suffix = ''
	outpath = './'
	savepath = './data/'

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os

def featurize(X_train):

    GROUPBY_AGGREGATIONS = [
        {'groupby': ['ip','day','hour'], 'select': 'minute', 'agg': 'nunique'},
        {'groupby': ['ip','day','hour'], 'select': 'minute', 'agg': 'count'},
        {'groupby': ['ip','day','hour','minute'], 'select': 'second', 'agg': 'nunique'},
        {'groupby': ['ip','day','hour','minute'], 'select': 'second', 'agg': 'count'},
        # {'groupby': ['ip','day','device'], 'select': 'click_time', 'agg': 'count'},
        # {'groupby': ['ip','day','device'], 'select': 'click_time', 'agg': 'nunique'},
        # {'groupby': ['ip','day','device'], 'select': 'app', 'agg': 'count'},
        # {'groupby': ['ip','day','device'], 'select': 'app', 'agg': 'nunique'},
        # {'groupby': ['ip','day','device'], 'select': 'os', 'agg': 'nunique'},
        # {'groupby': ['ip','day','device'], 'select': 'channel', 'agg': 'nunique'},
        # {'groupby': ['ip','day','hour','minute','second'], 'select': 'app', 'agg': 'nunique'},
        # {'groupby': ['ip','day','hour','minute','second'], 'select': 'app', 'agg': 'count'},
        # {'groupby': ['ip','day','hour','minute','second'], 'select': 'click_time', 'agg': 'count'},
        # {'groupby': ['ip','day','hour','minute','second'], 'select': 'os', 'agg': 'count'},
        # {'groupby': ['ip','day','hour'],'select':'click_time','agg':'count','addition':'max'}
    ]


    for spec in GROUPBY_AGGREGATIONS:
    
        # Name of the aggregation we're applying
        agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
    
        # Name of new feature
        new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
    
        # Info
        print("Grouping by {}, and aggregating {} with {}".format(
            spec['groupby'], spec['select'], agg_name
        ))
    
        # Unique list of features to select
        all_features = list(set(spec['groupby'] + [spec['select']]))
    
        # Perform the groupby
        gp = X_train[all_features]. \
            groupby(spec['groupby'])[spec['select']]. \
            agg(spec['agg']). \
            reset_index(). \
            rename(index=str, columns={spec['select']: new_feature})
        
        
        # Merge back to X_total
        if 'cumcount' == spec['agg']:
            X_train[new_feature] = gp[0].values
        else:
            X_train = X_train.merge(gp, on=spec['groupby'], how='left')
        
         # Clear memory
        del gp
        gc.collect()
    return X_train
	
def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_cumcount( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_mean( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_var( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def feat_ratio(df):
    print(df.columns)
    df['ip_day_hour_minuteR'] = df['ip_day_hour_count_minute']/df['ip_day_hour_nunique_minute']
    df['ip_day_hour_minute_secondR'] = df['ip_day_hour_minute_count_second']/df['ip_day_hour_minute_nunique_second']
    # df['ip_day_device_click_timeR'] = df['ip_day_device_count_click_time']/df['ip_day_device_nunique_click_time']
    # df['ip_day_device_appR'] = df['ip_day_device_count_app']/df['ip_day_device_nunique_app']
    # df['ip_day_device_appChannelR'] = df['ip_day_device_nunique_app']/df['ip_day_device_nunique_channel']
    # df['ip_day_hour_minute_second_appR'] = df['ip_day_hour_minute_second_nunique_app']/df['ip_day_hour_minute_second_count_app']
    return df

def do_attributed_prob(train_df, features):
    grouped = train_df.groupby(features)
    log_base = np.log(100000)
    new_feature = '_'.join(features) + "_attributed_rate"
    def rate_calculation(x):
        rate = x.sum() / float(x.count())
        conf = np.min([1, np.log(x.count()) / log_base])
        return rate * conf

    return train_df.merge(
            grouped['is_attributed']. \
                    apply(rate_calculation). \
                    reset_index(). \
                    rename(index=None, columns={'is_attributed': new_feature})[features + [new_feature]],
                on=features, how='left'
            )

debug=1 
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.2,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

    return (bst1,bst1.best_iteration)

def DO(frm,to,fileno):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    print('loading train data...',frm,to)
    train_df = pd.read_csv("./train.csv", parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

    print('loading test data...')
    if debug:
        test_df = pd.read_csv("./test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv("./test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

    len_train = len(train_df)
    train_df=train_df.append(test_df)
    
    del test_df
    gc.collect()
    
    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df['minute'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df['second'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    
    # gc.collect()
    # train_df = do_countuniq( train_df, ['ip'], 'channel', 'X0', 'uint8', show_max=True ); gc.collect()
    # train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app', 'X1', show_max=True ); gc.collect()
    # train_df = do_countuniq( train_df, ['ip', 'day'], 'hour', 'X2', 'uint8', show_max=True ); gc.collect()
    # train_df = do_countuniq( train_df, ['ip'], 'app', 'X3', 'uint8', show_max=True ); gc.collect()
    # train_df = do_countuniq( train_df, ['ip', 'app'], 'os', 'X4', 'uint8', show_max=True ); gc.collect()
    # train_df = do_countuniq( train_df, ['ip'], 'device', 'X5', 'uint16', show_max=True ); gc.collect()
    # train_df = do_countuniq( train_df, ['app'], 'channel', 'X6', show_max=True ); gc.collect()
    # train_df = do_cumcount( train_df, ['ip'], 'os', 'X7', show_max=True ); gc.collect()
    # train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app', 'X8', show_max=True ); gc.collect()
    # train_df = do_count( train_df, ['ip', 'day', 'hour'], 'ip_tcount', show_max=True ); gc.collect()
    # train_df = do_count( train_df, ['ip', 'app'], 'ip_app_count', show_max=True ); gc.collect()
    # train_df = do_count( train_df, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=True ); gc.collect()
    # train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour', 'ip_tchan_count', show_max=True ); gc.collect()
    # train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var', show_max=True ); gc.collect()
    # train_df = do_var( train_df, ['ip', 'app', 'channel'], 'day', 'ip_app_channel_var_day', show_max=True ); gc.collect()
    # train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_mean_hour', show_max=True ); gc.collect()
    # train_df = do_attributed_prob( train_df, ['ip']); gc.collect()
    # train_df = do_attributed_prob( train_df, ['app']); gc.collect()
    # train_df = do_attributed_prob( train_df, ['device']); gc.collect()
    # train_df = do_attributed_prob( train_df, ['os']); gc.collect()
    # train_df = do_attributed_prob( train_df, ['channel']); gc.collect()

    train_df = featurize(train_df)
    train_df = feat_ratio(train_df)

    print(train_df.columns)

    print('doing nextClick')
    predictors=[]
    
    new_feature = 'nextClick'
    filename='nextClick_%d_%d.csv'%(frm,to)

    if os.path.exists(filename):
        print('loading from save file')
        QQ=list(pd.read_csv(filename).values)
    else:
        D=2**26
        train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
            + "_" + train_df['os'].astype(str)).apply(hash) % D
        click_buffer= np.full(D, 3000000000, dtype=np.uint32)

        train_df['epochtime']= train_df['click_time'].astype(np.int64) // 10 ** 9
        next_clicks= []
        for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
            next_clicks.append(click_buffer[category]-t)
            click_buffer[category]= t
        del(click_buffer)
        QQ= list(reversed(next_clicks))
        train_df.drop(['epochtime','category'], axis=1, inplace=True)

        if not debug:
            print('saving')
            pd.DataFrame(QQ).to_csv(filename,index=False)
            

    train_df[new_feature] = pd.Series(QQ).astype('float32')
    predictors.append(new_feature)

    train_df[new_feature+'_shift'] = train_df[new_feature].shift(+1).values
    predictors.append(new_feature+'_shift')
    
    train_df.drop(['click_time', 'day'], axis=1, inplace=True)

    del QQ
    gc.collect()

    train_df = do_var( train_df, ['ip'], 'nextClick', 'ip_nextClick_var', show_max=True ); gc.collect()

    print("vars and data type: ")
    train_df.info()
    # train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    # train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    # train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

    print(train_df.columns)
    print(train_df['ip_nextClick_var'].head)


    target = 'is_attributed'
    # predictors.extend(['app','device','os', 'channel', 'hour', 
    #               'ip_tcount', 'ip_tchan_count', 'ip_app_count',
    #               'ip_app_os_count', 'ip_app_os_var',
    #               'ip_app_channel_var_day','ip_app_channel_mean_hour',
    #               'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8',
    #               'ip_attributed_rate', 'app_attributed_rate', 'device_attributed_rate', 
    #               'os_attributed_rate', 'channel_attributed_rate', 'ip_nextClick_var'])
    predictors.extend(['app', 'channel', 'click_id', 'device', 'ip', 'is_attributed', 'os',
       'hour', 'minute', 'second', 'ip_day_hour_nunique_minute',
       'ip_day_hour_count_minute', 'ip_day_hour_minute_nunique_second',
       'ip_day_hour_minute_count_second', 'ip_day_hour_minuteR',
       'ip_day_hour_minute_secondR', 'nextClick', 'nextClick_shift',
       'ip_nextClick_var'])
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    print('predictors',predictors)

    test_df = train_df[len_train:]
    val_df = train_df[(len_train-val_size):len_train]
    train_df = train_df[:(len_train-val_size)]

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    gc.collect()

    print("Training...")
    start_time = time.time()

    params = {
        'learning_rate': 0.20,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200 # because training data is extremely unbalanced 
    }
    (bst,best_iteration) = lgb_modelfit_nocv(params, 
                            train_df, 
                            val_df, 
                            predictors, 
                            target, 
                            objective='binary', 
                            metrics='auc',
                            early_stopping_rounds=30, 
                            verbose_eval=True, 
                            num_boost_round=1000, 
                            categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()

    if WHERE!='gcloud' and WHERE!='aws':
        print('Plot feature importances...')
        ax = lgb.plot_importance(bst, max_num_features=100)
        plt.show()

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
    if not debug:
        print("writing...")
        sub.to_csv('sub_it%d.csv'%(fileno),index=False,float_format='%.9f')
    print("done...")
    return sub

nrows=184903891-1
nchunk=25000000
val_size=2500000

frm=nrows-75000000
if debug:
    frm=0
    nchunk=100000
    val_size=10000

to=frm+nchunk

sub=DO(frm,to,0)
