# Experiments

## Parameters
```python
dice_exp = exp.generate_counterfactuals(
        example_df,
        total_CFs=1,
        verbose=True,
        min_iter=100,
        max_iter=2000,
        features_to_vary=[
            'amount',
            *activity_feature_names.tolist(),
            *resource_feature_names.tolist(),
        ],
    )
```

# A_APPROVED_COMPLETE

## Input

```python
### Activities
['A_SUBMITTED_COMPLETE',
 'A_PARTLYSUBMITTED_COMPLETE',
 'A_PREACCEPTED_COMPLETE',
 'A_ACCEPTED_COMPLETE',
 'A_FINALIZED_COMPLETE']

### Resources
['112', '112', '112', '11201', '11201']

### Amount
[15500.0]
```

## Prediction
```python
========================================Predict result========================================
| Predicted activity with highest probability (0.29) is "A_CANCELLED_COMPLETE" 
==============================================================================================
```

## Time
```18 min 50 sec for each round ```

### \subsection*{Change AMOUNT only}
```Not found```

### \subsection*{Change Resource only}
```Not found```

### \subsection*{Change Activity only}
```Not found```

### \subsection*{Change Amount and Resource}
```Not found```

### \subsection*{Change Amount and Activity}
```Not found```

### \subsection*{Change Activity and Resource}
```Not found```

### \subsection*{Change Activity, Resource and Amount}
```Not found```
