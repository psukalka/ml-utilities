# ML Utilities
Initially this repository will hold code snippets that I learn in AI-ML course. 
As my knowledge evolves, it might store other ML related utilities.

## Running script
python -m scripts.data_preprocessing

## Notes:
- Feature scaling shouldn't be applied on dummy variables (ex. one-hot encoded values) as they are already either 0 or 1 and would lose their meaning by scaling
- 5 Methods of Building a model:
    - All in : throw in all your variables
    - Backward Elimination
        - Select significance level (SL) to stay in the model
        - Fit the full model with all possible predictors
        - Consider the predictor with the highest P-value. If P-value > SL, go to step 4 else done
        - Remove the predictor
        - Fit the model without this variable
    - Forward Selection
        - Select a significance level (SL) to enter the model 
        - Fit all the simple regression models. Select the one with the lowest P-value
        - Keep this variable and fit all possible models with one extra predictor added to the ones you already have 
        - Consider the predictor with the lowest P-value. If P < SL go to step 3 else done
    - Bidirectional Elimination
        - Select a significance level to enter (SLENTER) and to stay (SLSTAY)
        - Perform the next step of forward selection (new variables must have P < SLENTER)
        - Perform all steps of Backward elimination (old variables must have P < SLSTAY)
        - Finish when no new variables can enter and exit
    - Score Comparison
        - Select a criterion of goodness of fit
        - Construct all possible regression models 
        - Select the one with the best criterion
    
    - We will use Backward Elimination in building models as it is fastest
- We don't need to apply feature scaling in multiple linear regression as the coefficients take care of it
- With sklearn library you don't need to worry about dummy variable trap 
- We don't need to work on BE technique as the sklearn model selects variables with highest P-values 
- Polynomial linear regression is called `linear` because of linearity of its coefficients
- In Support Vector Regression (SVR), we don't have coefficients with features that would balance out their scale, hence feature scaling is needed.
- Decision Tree Regression doesn't need feature scaling.