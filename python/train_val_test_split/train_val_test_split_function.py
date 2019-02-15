def train_val_test_split(X, y, test_size=0.2, random_state=0, over_sampling=False):
    """Gets input data and returns it splitted in three parts (training, validation and testing). Optionally, it can oversample the data.
    
    Parameters:
    -----------
    X: `pandas.DataFrame` or `pandas.Series`. Contains model features.
    y: `pandas.Series`. Contains the model target.
    test_size: `float`. test_size will be passed to sklearn.model_selection.train_test_split function.
    random_state: `integer`. random_state will be passed to sklearn.model_selection.train_test_split function.
    over_sampling: `boolean`. Indicates if data should be oversampled before returning the result.
    
    Returns:
    --------
    X_train = `pandas.DataFrame` or `pandas.Series`. Contains model features. Should be used on the model training phase.
    X_val = `pandas.DataFrame` or `pandas.Series`. Contains model features. Should be used on the model training phase.
    X_test = `pandas.DataFrame` or `pandas.Series`. Contains model features. Should be used for final model assessment.
    y_train = `pandas.Series`. Contains the model target. Should be used on the model training phase.
    y_val = `pandas.Series`. Contains the model target. Should be used on the model training phase.
    y_test = `pandas.Series`. Contains the model target. Should be used for final model assessment.
    """
    
    #Required function
    from sklearn.model_selection import train_test_split
    
    #Splits test from the rest
    X_model, X_test, y_model, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)  
    
    #Splits train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_model, y_model, test_size=test_size, random_state=random_state)  
    
    #If over_sampling flag is set to True, then data is oversampled
    if (over_sampling==True):
        #Required function
        from imblearn.over_sampling import SMOTE
        
        #Creates oversampled set
        balancer = SMOTE(kind='regular')
        X_resampled_train, y_resampled_train = balancer.fit_sample(X_train, y_train)
        X_resampled_val, y_resampled_val = balancer.fit_sample(X_val, y_val)
        X_resampled_test, y_resampled_test = balancer.fit_sample(X_test, y_test)
        
        #Overrides original variables to make returning easier
        X_train = X_resampled_train        
        y_train = y_resampled_train
        X_val = X_resampled_val
        y_val = y_resampled_val
        X_test = X_resampled_test
        y_test = y_resampled_test
        
    #Returns dataset
    return X_train,X_val,X_test,y_train,y_val,y_test
