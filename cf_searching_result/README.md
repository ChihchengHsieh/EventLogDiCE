
## Name of the file

{counterfactual(cf) or all_prediction(desired_df)}_{ground truth}_Amount_{amount argument value}_ReplaceAmount_{replace_amount argment value}_result.csv


## 3 ways to search and predict.

1. `amount = example_amount, replace_amount = None` (Find cases with same milestones and amount)

2. `amount = None, replace_aomunt = None` (Find cases with same milstones, and use the amount in dataset)

3. `amount = None, replace_amount = example_amonut` (Find cases with same milstones, and replace their amount by replace_amount)


## Algorithm to explain

```
Input:
BPI2012 activity milestones, milestones;
The activity to find counterfactual for, desired_activity;
Trace information, including activities, reosurces and amount, input_query;

1. milstone trace <- only keep the activties that are not milestones in the trace.
2. potential conterfactuals <- Find the cases containing milestone trace in training set.
3. if amount is fixed:
        only keep the cases has the same amount as input query for potential counterfactuals.

4. if replace amount:
        replace amount for all cases in potential counterfactual by the amount of input query.

5. make prediction for all cases in potential counterfactuals 

6. Check the prediction to indentify counterfactual.
```
