
#Load in packages
from gpboost.engine import train
import pandas as pd
import numpy as np
import gpboost as gpb
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn.model_selection import GroupShuffleSplit

#change options and set seed
pd.set_option("display.max_rows", None, "display.max_columns", None)
np.random.seed(59)

#load in data & check
pbp_data = pd.read_csv("C:/Users/Joe/Desktop/Coordinator_NFL_Final.csv")
pbp_data.head(5)

#select our target and variables of interest, going to stick with non spread adjusted WP in the model
#reason behind the non spread is my theory that OC's don't view the game through the spread lense
modeling_data = pbp_data[["yardline_100", 
                          "qtr",
                          "half_seconds_remaining", 
                          "ydstogo",
                          "down",
                          "shotgun",            
                          "score_differential",
                          "posteam_timeouts_remaining",
                          "defteam_timeouts_remaining",
                          "wp", 
                          "O_Coordinator",
                          "pass"]]

#Objectives of this analysis
#1. Develop a probability of pass model that incorporates coordinator random effects
#2. Examine if certain DC's are passed or rushed against more than their peers

# Model development
# Step one is to split into training and test, I am 

train_final  = pd.DataFrame()
test_final   = pd.DataFrame()
Coordinators = modeling_data['O_Coordinator'].unique()
length = len(Coordinators)
for x in range(length):
    Coordinator_X = Coordinators[x]
    Data_Filtered = modeling_data[modeling_data['O_Coordinator'] == Coordinator_X]
    train, test   = train_test_split(Data_Filtered, test_size= .3, random_state=94)
    train_final   = train_final.append(train)
    test_final    = test_final.append(test)

# Set likelihood function to use
likelihood = "bernoulli_logit"

# Define random effects model (OC) & inser the likelihood
gp_model = gpb.GPModel(group_data=train_final[['O_Coordinator']], likelihood=likelihood)

# Create dataset for gpb.train
data_train = gpb.Dataset(data=train_final[[
     "yardline_100", 
     "qtr",
     "half_seconds_remaining",
     "ydstogo",
     "down",
     "shotgun",        
     "score_differential",
     "posteam_timeouts_remaining",
     "defteam_timeouts_remaining",
     "wp"]],
     label=train_final['pass']) #pass is our target so we set that as the label

# Set model parameters
params = {'objective': 'binary', 
          'verbose': 0,
          'num_leaves': 2**10 }

# Grid for model to search through
param_grid = {'learning_rate': [0.5,0.1,0.05,0.01,.2], #eta
                'max_depth': [1,3,5,10,20], #max_depth
                'min_sum_hessian_in_leaf': [2,8,15,20,25,40], #min_child_weight
                'bagging_fraction': [.1,.25,.35,.55,.75], #subsample
                'feature_fraction': [.1,.25,.35,.55,.75]} #colsample_bytree

# Grid search
opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid,
                                             params=params,
                                             num_try_random=10, #40 rounds
                                             nfold=5, #5 fold cross validation
                                             gp_model=gp_model, #random effects
                                             use_gp_model_for_validation=True,
                                             train_set=data_train, #training set
                                             verbose_eval=1,
                                             num_boost_round=1000, 
                                             early_stopping_rounds=25, #stop at 25 if no improvement
                                             seed=1,
                                             metrics='binary_logloss') #log loss as eval metric

# Print best options
print("Best number of iterations: " + str(opt_params['best_iter']))
print("Best score: " + str(opt_params['best_score']))
print("Best parameters: " + str(opt_params['best_params']))

# Reevaluate parameters
param_grid = {'learning_rate': [0.2031687, 0.1564337, 0.1795145, 0.2189100, 0.2558462], 
                'max_depth': [3,4,5,6,7,8],
                'min_sum_hessian_in_leaf': [30,35,40,45,50],
                'bagging_fraction': [.25,.3,.35,.4,.45],
                'feature_fraction': [.15,.2,.25,.3,.35]}

# Same grid search with new parameters
opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid,
                                             params=params,
                                             num_try_random=15,
                                             nfold=4,
                                             gp_model=gp_model,
                                             use_gp_model_for_validation=True,
                                             train_set=data_train,
                                             verbose_eval=1,
                                             num_boost_round=1000, 
                                             early_stopping_rounds=15,
                                             seed=1,
                                             metrics='binary_logloss')
print("Best number of iterations: " + str(opt_params['best_iter']))
print("Best score: " + str(opt_params['best_score']))
print("Best parameters: " + str(opt_params['best_params']))

# Train with best parameters
params = { 'objective': 'binary',
            'learning_rate': 0.2, 
            'max_depth': 5,
            'min_sum_hessian_in_leaf': 40,
            'bagging_fraction': 0.35,
            'feature_fraction': 0.25,
            'verbose': 0 }

# Train GPBoost model
bst = gpb.train(params=params,
                train_set=data_train,
                gp_model=gp_model,
                num_boost_round=335,
                valid_sets=data_train)

# Estimated random effects model
gp_model.summary()

# Check importance
gpb.plot_importance(bst)

# Define random effects model (OC) 
group_test = test_final[["O_Coordinator"]]

# Create dataset for gpb.train
data_test = test_final[[
     "yardline_100", 
     "qtr",
     "half_seconds_remaining",
     "ydstogo",
     "down",
     "shotgun",        
     "score_differential",
     "posteam_timeouts_remaining",
     "defteam_timeouts_remaining",
     "wp"]]

# Make predictions on the test set
pred = bst.predict(data=data_test,
                    group_data_pred= group_test,
                    raw_score=False)

# Add predictions to dataframe of variables
Probs = pd.DataFrame(data = pred['response_mean'])
test_final.reset_index(drop=True, inplace=True)
Probs.reset_index(drop=True, inplace=True)
df = pd.concat( [test_final, Probs], axis=1) 
df.to_excel('test.xlsx')









# Best params so far
params = { 'objective': 'binary',
            'learning_rate': 0.2, 
            'max_depth': 5,
            'min_sum_hessian_in_leaf': 40,
            'bagging_fraction': 0.35,
            'feature_fraction': 0.25,
            'verbose': 0 }


#save model
bst.save_model('model.json')
