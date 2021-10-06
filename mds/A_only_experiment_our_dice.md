# Experiments

## Parameters
```python
cf_out = dice.run_pls(
    ## Input
    example_amount_input.numpy(),
    example_idx_activities_no_tag,
    example_idx_resources_no_tag,
    desired_vocab = "A_FINALIZED_COMPLETE",
    
    ## Weight
    class_loss_weight = 1,
    scenario_weight = 2e+2,
    distance_loss_weight = 1e-8,
    cat_loss_weight = 1e-3,
    
    ## Training parameters
    scenario_threshold = 0.3,
    scenario_ask_to_continue=True,
    max_iter=1000,
    lr=0.5,

    ## Updating fields
    updating_amount_cf=False,
    # updating_resource_cf=False,
    updating_activity_cf=False,
    
    ## Options
    use_valid_cf_only=False,
    use_sampling=True,
    class_using_hinge_loss=False,
    scenario_using_hinge_loss=False,
    use_clipping=False, 
)
```

# A_APPROVED_COMPLETE

## Input

```python

====================Input Amount====================
| [15500.] 
====================================================

====================Input Activities====================
| ['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'A_ACCEPTED_COMPLETE', 'A_FINALIZED_COMPLETE'] 
========================================================

====================Input Resource====================
| ['112', '112', '112', '11201', '11201'] 
======================================================
```

## Prediction

## Time
```18 min 50 sec for each round ```

### \subsection*{Change AMOUNT only}
```Not found```

### \subsection*{Change Resource only}
```Not found```

### \subsection*{Change Activity only} (found but not valid trace)
```python
====================Valid CF Amount====================
| 15500.0 
=======================================================

====================Valid CF Activities====================
| ['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'A_ACCEPTED_COMPLETE'] 
===========================================================

====================Valid CF Resource====================
| ['112', '112', '112', '11201', '11201'] 
=========================================================

====================Valid CF scenario output====================
| [0.8 1.  1.  1.  1.  1. ] 
================================================================
```

### \subsection*{Change Amount and Resource}
```Not found```

### \subsection*{Change Amount and Activity} (found but not valid trace)
```python
====================Valid CF Amount====================
| 15491.5 
=======================================================

====================Valid CF Activities====================
| ['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'A_ACCEPTED_COMPLETE'] 
===========================================================

====================Valid CF Resource====================
| ['112', '112', '112', '11201', '11201'] 
=========================================================

====================Valid CF scenario output====================
| [0.8 1.  1.  1.  1.  1. ] 
================================================================
```

### \subsection*{Change Activity and Resource} (found but not valid)
```
====================Valid CF Amount====================
| 15500.0 
=======================================================

====================Valid CF Activities====================
| ['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'A_ACCEPTED_COMPLETE'] 
===========================================================

====================Valid CF Resource====================
| ['112', '112', '112', '11181', '10935'] 
=========================================================

====================Valid CF scenario output====================
| [0.8 1.  1.  1.  1.  1. ] 
================================================================
```

### \subsection*{Change Activity, Resource and Amount} (found but not valid trace)
```
====================Valid CF Amount====================
| 15497.0 
=======================================================

====================Valid CF Activities====================
| ['A_PARTLYSUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'A_APPROVED_COMPLETE'] 
===========================================================

====================Valid CF Resource====================
| ['112', '112', '10629', '10972', '11201'] 
=========================================================

====================Valid CF scenario output====================
| [0.8 0.  0.  0.  0.2 0.5] 
================================================================
```
