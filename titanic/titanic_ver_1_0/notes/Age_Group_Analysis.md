# Child Group Analysis
I first suspected the survival disparity based on age can easily influence outcome, and some age interval based EDA didnt dissapoint me.

## Hypothesis on observation
An interesting observation was Child survival was not linear with age. The youngest children had high survival, older children around 7-13 had a noticeably worse survival profile, and early teens recovered somewhat, possibly because gender/social category began to matter more.

![ 5 year Child AgeGroup Interval](../train_visual_plots/Child%20Group%20Plots/EDA_based_child_fitting.png)

## Absolute 2 year interval between 0 to 16:

![2 year interval plot](../train_visual_plots/Child%20Group%20Plots/2_year_interval_child_group.png)

Specific to 2 interval Gap:
0-2      very high survival
2-4      mixed survival
4-6      high survival
6-8      50/50-ish
8-10     bad survival
10-12    very bad survival
12-14    very high survival, but tiny sample
14-16    mixed

## Varying intervals till decent sample size yet accounting intricacies of specific age group

![EDA based intervals and sample fitting](../train_visual_plots/Child%20Group%20Plots/EDA_based_child_fitting.png)

0-2      high survival
2-3      weaker survival
3-5      high survival
5-7      decent/high survival
7-10     weak survival
10-13    weak survival
13-16    improves again

## Overall
0-2 and 3-7 look protected
7-13 looks unusually vulnerable
13-16 recovers somewhat