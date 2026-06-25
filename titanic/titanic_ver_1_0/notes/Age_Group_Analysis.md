# Child and Sex-Specific Age Group Analysis

I suspected that age-based survival differences could meaningfully influence the outcome, and the initial EDA supported that idea.

## Child Age Survival Pattern

One interesting observation was that child survival was not linear with age. The youngest children had high survival, children around ages 7-13 had a noticeably weaker survival profile, and early teenagers seemed to recover somewhat. This may be because gender and social category started to matter more strongly at older child ages.

![5-year child age group interval](../train_visual_plots/Child%20Group%20Plots/EDA_based_child_fitting.png)

## Two-Year Child Age Intervals

![2-year interval plot](../train_visual_plots/Child%20Group%20Plots/2_year_interval_child_group.png)

Using absolute 2-year intervals from age 0 to 16, the pattern was:

- `0-2`: very high survival
- `2-4`: mixed survival
- `4-6`: high survival
- `6-8`: roughly balanced survival/death
- `8-10`: weak survival
- `10-12`: very weak survival
- `12-14`: very high survival, but with a tiny sample
- `14-16`: mixed survival

## EDA-Based Child Age Intervals

![EDA-based intervals and sample fitting](../train_visual_plots/Child%20Group%20Plots/EDA_based_child_fitting.png)

To reduce volatility while still preserving meaningful age changes, I also tested variable-width age intervals:

- `0-2`: high survival
- `2-3`: weaker survival
- `3-5`: high survival
- `5-7`: decent/high survival
- `7-10`: weak survival
- `10-13`: weak survival
- `13-16`: improved survival again

This supported the idea that child age should not be treated as one simple category.

## Sex-Based Survival Difference Among Children

Even among children, survival rates differed noticeably by sex. This suggested that age-based survival patterns should not be analyzed independently from sex.

![Sex-based differentiation in child survival rate](../train_visual_plots/Child%20Group%20Plots/Sex_based_child_group.png)

## Adult Age Pattern

The adult group had more samples and showed less volatile survival changes compared with the child group. However, the adult survival pattern still differed by sex, so male and female age bins were analyzed separately for fairness and better signal capture.

![Less volatile changes in adult group](../train_visual_plots/Adult%20Group%20Plots/Adult%205%20interval%20Plot.png)

## Statistical Sex-Specific Age Binning

Instead of using fixed manual age ranges, I created `Sex_Age_Group` using a data-driven binning method.

Titanic survival patterns are strongly affected by both age and sex, so I generated separate age bins for male and female passengers. I first removed rows with missing age values, then split the data by `Sex`. Within each sex group, I further separated passengers into children and adults using age 16 as the boundary.

For each sex, I trained two small `DecisionTreeClassifier` models:

- A child-age tree for passengers aged 16 or below.
- An adult-age tree for passengers older than 16.

The child tree used a smaller `min_samples_leaf`, allowing it to detect finer survival changes among younger passengers. The adult tree used a larger `min_samples_leaf`, forcing broader and more stable bins for adults, where small age differences were more likely to be noise.

After training each tree, I extracted the learned age split thresholds from the tree structure. These thresholds became the bin edges for that sex. I then combined:

- `0.0` as the starting age
- Child-tree split points
- `16.0` as the child/adult boundary
- Adult-tree split points
- The maximum observed age plus one as the final upper boundary

This produced different statistical age bins for male and female passengers.

```python
female_age_bins = [
    0.0, 1.5, 3.5, 5.5, 7.5, 12.0, 14.8, 15.5,
    16.0, 21.5, 24.5, 28.5, 32.2, 40.5, 48.5, 64.0
]

male_age_bins = [
    0.0, 1.0, 1.5, 2.5, 3.5, 6.5, 8.5, 9.5, 13.0, 15.5,
    16.0, 20.2, 24.8, 27.5, 30.8, 32.2, 34.8, 36.2,
    47.5, 53.0, 81.0
]