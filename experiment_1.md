# Experiments

# A_ACCEPTED_COMPLETE

## Input 

```python
====================Input Amount====================
| [15500.] 
====================================================

====================Input Activities====================
| ['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'W_Completeren aanvraag_COMPLETE'] 
========================================================

====================Input Resource====================
| ['112', '112', '112', '11180', '11201'] 
======================================================
```

## Prediction

```python
====================Model Prediction====================
| Prediction: [W_Completeren aanvraag_COMPLETE(22)] | Desired: [A_ACCEPTED_COMPLETE(3)] 
========================================================

====================Counterfactual Process====================
| [0] ==========> [1] 
==============================================================
```

### \subsection*{Change AMOUNT only}
```Not found```

### \subsection*{Change Resource only}
```python
====================Valid CF Amount====================
| 15500.0 
=======================================================

====================Valid CF Activities====================
| ['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'W_Completeren aanvraag_COMPLETE'] 
===========================================================

====================Valid CF Resource====================
| ['10125', '10124', '10779', '10914', '10789'] 
=========================================================

====================Valid CF scenario output====================
| [0.7 0.  0.  0.  0.  0. ] 
================================================================
```

### \subsection*{Change Activity only}
```Not found```

### \subsection*{Change Amount and Resource}
```python
====================Valid CF Amount====================
| 15409.707 
=======================================================

====================Valid CF Activities====================
| ['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'W_Completeren aanvraag_COMPLETE'] 
===========================================================

====================Valid CF Resource====================
| ['10188', '10859', '11269', '11339', '11029'] 
=========================================================

====================Valid CF scenario output====================
| [0.7 0.  0.  0.  0.  0. ] 
================================================================
```

### \subsection*{Change Amount and Activity}

```python
====================Valid CF Amount====================
| 14931.504 
=======================================================

====================Valid CF Activities====================
| ['O_CREATED_COMPLETE', 'O_CREATED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'O_DECLINED_COMPLETE', 'A_ACTIVATED_COMPLETE'] 
===========================================================

====================Valid CF Resource====================
| ['112', '112', '112', '11180', '11201'] 
=========================================================

====================Valid CF scenario output====================
| [0.7 0.  0.1 0.5 0.7 0.5] 
================================================================
```

### \subsection*{Change Activity and Resource}
```Not found```

### \subsection*{Change Activity, Resource and Amount}
```Not found```


# A_FINALIZED_COMPLETE

## Input 
```python
====================Input Amount====================
| [15500.] 
====================================================

====================Input Activities====================
| ['A_SUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'W_Completeren aanvraag_COMPLETE', 'A_ACCEPTED_COMPLETE'] 
========================================================

====================Input Resource====================
| ['112', '112', '112', '11180', '11201', '11201'] 
======================================================
```

## Prediction

```python
====================Model Prediction====================
| Prediction: [O_SELECTED_COMPLETE(17)] | Desired: [A_ACCEPTED_COMPLETE(3)] 
========================================================

====================Counterfactual Process====================
| [0] ==========> [1] 
==============================================================
```

### \subsection*{Change AMOUNT only}
```Not found```

### \subsection*{Change Resource only}
```Not found```

### \subsection*{Change Activity only}

```python
====================Valid CF Amount====================
| 15500.0 
=======================================================

====================Valid CF Activities====================
| ['A_PARTLYSUBMITTED_COMPLETE', 'A_PARTLYSUBMITTED_COMPLETE', 'A_ACTIVATED_COMPLETE', 'A_ACTIVATED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'A_REGISTERED_COMPLETE'] 
===========================================================

====================Valid CF Resource====================
| ['112', '112', '112', '11180', '11201', '11201'] 
=========================================================

====================Valid CF scenario output====================
| [0.7 0.  0.5 0.9 1.  0.8 0.4] 
================================================================
```

### \subsection*{Change Amount and Resource}

```Not found```

### \subsection*{Change Amount and Activity}

```python
====================Valid CF Amount====================
| 15225.252 
=======================================================

====================Valid CF Activities====================
| ['A_FINALIZED_COMPLETE', 'A_ACTIVATED_COMPLETE', 'A_ACTIVATED_COMPLETE', 'A_ACTIVATED_COMPLETE', 'A_PREACCEPTED_COMPLETE', 'A_APPROVED_COMPLETE'] 
===========================================================

====================Valid CF Resource====================
| ['112', '112', '112', '11180', '11201', '11201'] 
=========================================================

====================Valid CF scenario output====================
| [0.7 0.  0.1 0.4 0.6 0.5 0.2] 
================================================================
```

### \subsection*{Change Activity and Resource}
```Not found```

### \subsection*{Change Activity, Resource and Amount}
```Not found```