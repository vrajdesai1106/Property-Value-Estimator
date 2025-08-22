## AI Property Value Estimator

### Objective:
- The project predicts the market price of a property based on its features such as size, bedrooms, bathrooms, location attribute, and condition. This tool helps real estate professionals, buyers, and sellers quickly estimate property values using historical data.

### Project Files:
1. Dataset : USA Housing Dataset from Kaggle

    Key Features:
    * Date : Date of the property listing.
    * Price : Target variable, property price
    * Property attribute : bedrooms , bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement.
    * construction and reovation : yr_built, yr_renovated.
    * Location : street, city, statezip, country.

2. benchmark.py : Benchmarking Models.
    
    * Libraries: pandas, sklearn.model_selection, lazypredict

    * Process:
        1. Load dataset and separate numeric features as predictors.
        2. Split data into training and testing sets (80/20 split).
        3. Use LazyRegressor to train multiple regression algorithms and generate a leaderboard.
        4. Identify the best-performing model based on metrics like R² and RMSE.
    * Outcome: Provides a comparison of different regression models to guide final model selection.

3. main.py : Model Training

    * Libraries: pandas, sklearn.model_selection, sklearn.linear_model, pickle

    * Process:
        1. Keep only numeric columns and define price as the target.
        2. Split data into training and test sets (optional validation).
        3. Train the chosen regression model (PassiveAggressiveRegressor) on the training data.
        4. Save the trained model as model.pkl for use in the interactive app.
    * Outcome: A finalized regression model ready for deployment.

3. app.py : Gradio Interactive application
    * Libraries: gradio, pandas, pickle

    * Functionality:
        1. Load the trained model.pkl.
        2. Collect property details via Gradio input widgets (sliders, dropdowns, numeric inputs).
        3. Process inputs into a DataFrame compatible with the trained model.
        4. Predict property price and display it in a formatted output box.
        5. Styled interface for better user experience with background color and custom button colors.

    * Key Features of App:
        1. Real-time price estimation
        2. User-friendly layout with sliders and numeric inputs.
        3. Immediate results with formatted price display.

### How To Run
1. Install required Python packages: 

         pip install pandas scikit-learn lazypredict gradio
2. Place the dataset in Dataset/USA Housing Dataset.csv
3. Run benchmark.py to check model performance across algorithms.
4. Train and save the final model using:

        python main.py
5. Launch the interactive Gradio app:

        python app.py
6. Input property details in the interface to get estimated prices. 

### Notes and Observations:
* Benchmarking: LazyRegressor simplifies model comparison and selection.
* Model Choice: PassiveAggressiveRegressor selected based on benchmarking results.
* Input Format: Ensure numeric features match the training dataset for accurate predictions.
* App Design: Gradio blocks and styling improve usability and visual appeal.