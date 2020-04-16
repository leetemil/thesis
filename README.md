# MASTER'S THESIS
Master's Thesis Git Repository.

## Debbie Marks Questions
- skip connections
- model diagram
- tanh, sigma split activations, what up?
- spearman rho calculation? is it the same as deep sequence?
- data: how did you exactly get these? How can we reproduce the process. How is it different from deep sequence, i.e. can this difference explain the change in performance?
- it seems that activation is set to "relu", but the ```_nonlinear``` function only implements elu. Effectively, this seems to cause no activation.

## 2020-01-28 Wouter Questions
- Variational Auto Encoders: what exactly is forced to be unit Gaussian?
- Metrics and benchmarks
- Papers
- Plan until next week
- Cross entropy loss vs. negative log likelihood loss, and applying activation sigmoid before. what makes sense?
- aligned data; many gaps, missing amino acids? What is alignment exactly.
- Can we overcome alignment/Do something else?
- Small protein families, how do we tackle those?
- CNN's?
- co-volution between domains?
- First step: Explore space. (1) Pick a protein. (2) Encode it to get mean, logvar for the distribution of the protein. (3) Sample from the distribution. (4) Decode samples.

## Weekly Schedule

|               | Monday | Tuesday        | Wednesday | Thursday | Friday        |
|:-------------:|--------|----------------|-----------|----------|---------------|
| 08:00 - 09:00 |        |                |           |          |               |
| 09:00 - 10:00 |        |                |           |          |               |
| 10:00 - 11:00 |        |                |           |          |               |
| 11:00 - 12:00 | Lunch  | Lunch          | Lunch     | Lunch    | write all day |
| 12:00 - 13:00 |        |                |           |          |               |
| 13:00 - 14:00 |        | Wouter Meeting |           |          |               |
| 14:00 - 15:00 |        |                |           |          |               |
| 15:00 - 16:00 |        |                |           |          |               |

## 2020-01-21 Notes:
- 90 total (work)days.
- Writing ~2 pages each (4 total) every Friday equals ~72 pages in the end, which can then be cut down to ~60 pages.
- Sections that probably can be written right away: Proteins, Deep Learning, Representations, Autoencoders, Sequence Learning.
- By Feb 11th: 12 pages

Fra mødet: første tanke, sekventiel model, med VAE. Måske i stedet dele af proteinrummet i stedet for 
- der er nogle tasks de er dårlige til. ss cp.
