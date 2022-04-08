# Cluster Project - Zillow
<h2>Objectives:</h2>

- Document code, process, findings, and key takeaways in a Jupyter Notebook Final Report.
- Creation of modules the make that process repeatable and easy to follow.
- Ask exploratory questions of the data that will help understand more about the attributes and driveing the Zestimate(logerror).
- Construct a model that perdicts logerror for single family properties using clustering and regression techniques.
- Make recommendations to a data science team about how to improve predictions.
- Deliver a Report in the form of a 5 minute presentation.
- Answer any questions the audience may have.

<h2>Buisness Goals:</h2>

- Construct an ML Regression model that predict Zestimate(logerr) values of Single Family Properties using attributes of the properties and clustering.
- Find the key drivers of property logerror.
- Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.
- Make recommendations on what works or doesn't work in prediction these homes' values.

<h2>Audience:</h2>

- Target audience is Zillow data science team.

<h2>Deliverables:</h2>

- Up-to-date github repository with code, notebooks, and data. link: https://github.com/craigcalzado/clustering-project
- Live presentation of the project. 
- README.md file with project description.
- Final report in the form of a Jupyter Notebook.
- Necessary modules for project repoduction. Acquire & Prepare Modules (.py)
- Instructions on how to replicate the project.

<h2>Context/Scenario:</h2>

We want to create a better model that is able to predict the property value with better than the zestimate's prediction of Single Family Properties that had a transaction during 2017.

We have a model already, but we are hoping your insights can help us improve it. I need recommendations on a way to make a better model. Maybe you will create a new feature out of existing ones that works better, try a non-linear regression algorithm, or try to create a different model for each county. Whatever you find that works (or doesn't work) will be useful. Given you have just joined our team, we are excited to see your outside perspective.

<h2>Data Dictionary</h2>

| Attribute | Definition | Data Type | Additional Info |
| --- | --- | --- | --- |
| parcelid | Unique identifier for each property | int64 | dropped after 'logerror' and 'transactiondate' concat |
| taxvaluedollarcnt | Total property tax assessed value | float64 |renamed to 'value' |
| taxamount | Total tax amount for the property | float64 | dropped after 'tax_rate' |
| yearbuilt | Original construction date | float64 | changed to 'year' |
| lotsizesquarefeet | Size of property in square feet | float64 | changed to lotsqft |
| bedroomcnt | Number of bedrooms | float64 | changed to bedrooms |
| calculatedfinishedsquarefeet | Size of property in square feet | float64 | changed to sqft |
| bathroomcnt | Number of bathrooms | float64 |changed to bedroom |
| taxamount | Total tax amount for the property | float64 | dropped after 'tax_rate' to prevent data leakage |
| logerror | Difference between property tax and property value | integer | |
| fips | County FIPS code | integer | float64 |
| transactiondate | Date of transaction | object | dropped due to no use |
| tax_rate | Tax rate for the county | float64 | dropped after 'taxamount' |
| total_rooms | Total number of bed and bathrooms | float64 | comibination of bathroomcnt and bedroom cnt |
| dollar_per_sqft | Average dollar per square foot | float64 | |
| lot_size | Size of property in square feet | float64 | changed to lotsqft |

<h2>Questions:</h2>

- What is the distribution of each variable?
- Does the logerror differ by home age?
- Does the logerror differ by tax rate?
- Does the logerror differ by tax amount per sqft?
- Does the logerror differ by location?
- Is there a relationship between the logerror and amount of rooms?
- Is there a relationship between the logerror and age of property?
- Is there a relationship between the logerror and tax rate?
- If we control for age, does the logerror differ by location?
- If we control for tax rate, does the logerror differ by location?
- If we control for cost per sqft, does the logerror differ by location?
- If we control for location, does the logerror differ by age?
- If we control for location, does the logerror differ by tax rate?

<h2>Hypotheses;</h2>

- Alpha = .05 (95% confidence level)

<h3>Hypothesis #1 Does the logerror differ by home age?</h3>

-  $H_0:$ There is no correlation between logerror and the home age.
-  $H_a:$ There is a correlation between logerror and the home age.

<h3>Hypothesis #2 Does the logerror differ by cost per sqft?</h3>

-  $H_0:$ There is no correlation between logerror and the cost per sqft.
-  $H_a:$ There is a correlation between logerror and the cost per sqft.

<h3>Hypothesis #3 Is there a relationship between the logerror and amount of rooms</h3>

-  $H_0:$ There is no correlation between logerror and the amount of rooms.
-  $H_a:$ There is a correlation between logerror and the amount of rooms.

<h3>Hypoesis #4 Is there a relationship between cluster1 and logerror</h3>

-  $H_0:$ There is no correlation between cluster1 and the logerror.
-  $H_a:$ There is a correlation between cluster1 and the logerror. 


<h3>Hypoesis #5 Is there a relationship between cluster2 and logerror</h3>

-  $H_0:$ There is no correlation between cluster2 and the logerror.
-  $H_a:$ There is a correlation between cluster2 and the logerror. 

<h3>Hypoesis #6 Is there a relationship between cluster3 and logerror</h3>

-  $H_0:$ There is no correlation between cluster3 and the logerror.
-  $H_a:$ There is a correlation between cluster3 and the logerror.

<h4>All the nulls were rejected on all hypothesis tests. The data is not linearly separable. Pearsonr correlation test and T-test were ran.</h4>

<h2>Executive Summary/Conclusion and Recommendations</h2>

Discovery:

- There were six drivers for error in the zillow zestimate.
    - age_in_years
    - total_rooms
    - dollars_per_sqft
    - calculatedfinishedsquarefeet
    - cluster1('age_in_years' vs. 'dollars_per_sqft')
    - cluster2('total_rooms' vs. 'calculatedfinishedsquarefeet')

- Modeling 
    - 3rd degree Polynomial Regression model performed best with 1% improvement over the baseline.


<h2>Reproduction</h2>
github: https://github.com/craigcalzado/cluster-project
Import modules:
    - wrangle.py (data wrangling)
Run zillow_project.ipynb


<h2>Project Summary</h2>

I wanted to create a regression model that predict prices better than logerror values of Single Family Properties using attributes of the properties. I needed to find the key drivers of property value to deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was. I needed to make recommendations on what works or doesn't work in prediction these homes' values.
The tools I used to create this model were: 
- Pandas
- Numpy
- Sklearn
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
I was able to create 3 degree polyregression model that predict the value with decreased logerror of Single Family Properties using age_in_years, total_room,dollars_per_sqft, calculatedfinishedsquarefeet, cluster1('age_in_years' vs. 'dollars_per_sqft'), and cluster2('total_rooms' vs. 'calculatedfinishedsquarefeet'). I set the baseline at the mean value of the logerror for  Single Family Properties which was .165223 . I used the 3rd degree polynomial model to predict the logerror value of .1611315 which is a .05% improvement.
