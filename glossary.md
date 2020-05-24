## This information is collected from Slack discussions and Q&A.

### Keywords: Energy allocation, optimization, constraints

**Question**
*Jonathan Whitaker*
*[Link to thread](https://prohackworkspace.slack.com/archives/C011W8E2UAX/p1589970281276900?thread_ts=1589964506.264400&cid=C011W8E2UAX)*

**For the optimization part I'm allocating in proportion to (potential for increase **2), clipping to 100 and fudging slightly to get 10% to LE galaxies. So nothing fancy - this seemed OK and let me compare prediction approaches.
I think the optimization part does affect the score quite a lot - I've seen a few folks with very low RMSE in testing, but scores still ~0.1 on the leaderboard. Would be interested to hear if anyone has tips on it :slightly_smiling_face:**

**Answer**
*Jonathan Whitaker*

	Not worth keeping secret - I'm sure it can be improved!! But here's the code for what it's worth:
	test['pred_y_cat2'] = model.predict(test[cols].fillna(0))
	test['potential_increase'] = -np.log(test['pred_y_cat2']+0.01)+3
	ss = pd.DataFrame({
	    'Index':test.index,
	    'pred':test['pred_y_cat2'],
	    'opt_pred':0,
	    'eei':test['existence expectancy index'], # So we can split into low and high EEI galaxies
	    'potential_increase':test['potential_increase'],
	    'p2':test['potential_increase']**2
	})
	# Weight by p2
	ss['opt_pred'] = (ss['p2']*50000/ss['p2'].sum()).clip(0, 100)
	# Rescale to get 5k to low-EEI
	low_sum = ss.loc[ss.eei < 0.7, 'opt_pred'].sum()
	ss.loc[ss.eei < 0.7, 'opt_pred'] = (ss.loc[ss.eei < 0.7, 'opt_pred']*5001/low_sum).clip(0, 100) # Fudge it above 5000
	high_sum = ss.loc[ss.eei >= 0.7, 'opt_pred'].sum()
	ss.loc[ss.eei >= 0.7, 'opt_pred'] = (ss.loc[ss.eei >= 0.7, 'opt_pred']*44600/high_sum).clip(0, 100) # Tweak to keep total below 50k
	increase = (ss['opt_pred']*ss['potential_increase']**2)/1000
	print(sum(increase), ss.loc[ss.eei < 0.7, 'opt_pred'].sum(), ss['opt_pred'].sum())
	ss[['Index', 'pred', 'opt_pred']].to_csv('submission.csv', index=False)


**Question**
*Jonathan Whitaker*

**Some minor notes:
The slides shared specify the need for a 'pred_opt' column, but you need 'opt_pred' instead or you'll get an error submitting.
The submission upload asks for PDF, EXL or DOC, but needs a CSV by the look of it.
And then question: For the optimization part, are we trying to get the maximum increase in energy compound index? Is the RMSE calculated on the difference between the proposed allocations and the 'optimum' values?
And final question: Are we allowed to share code? I can share a notebook to help folks get started if that's OK :slightly_smiling_face:**

**Answer**
*Dmitriy Sholomitskiy*

	I agree that "RMSE on allocation" sounds a bit wierd. I guess participants can get slightly different allocations but with almost the same high objective function score, and RMSE might be high.

*Aleksandr Finagin*

	@Jonathan Whitaker yes, the goal is to optimize total index increase by allocating energy
	rmse is calculated on allocated energy and optimal allocation values


**Question**
*Minawi*

**Guys, I have this problem that is bugging me and I wanted to know how u approached it.
Are the 50k energy distributed per galactic year or per unique galaxy since there are duplicates and this part is very vague and not clear.**

**Answer**
*Adilet Gaparov*
	
	You need to allocate 50k energy across all 890 rows of test set. Those energy allocation depends on your prediction of the index, since the goal of energy allocation is to maximize total increase in [well-being] index.


**Question**
*Abzal*

**Hey, can condition "at least 10% of the total energy available in the foreseeable future" and "no more than 100 zillion DSML" contradict each other? Otherwise, I keep getting either 100 (mostly for <0.7 countries) and 0s for >0.7**

**Answer**
*Adilet Gaparov*

	sum(Energy received by ALL galaxies with <0.7) >= 10 % * 50000 = 5000


**Question**
*Tofig Nazarbayov*

**Hey, guys! Would be grateful if someone clarifies few things about optimization problem. As I got, we have to maximize likely increase in the index, by optimizing energy allocation values. So we have 2 unknowns here - energy allocation values and index under logarithmic function. Energy value is the one that we will optimize, by maximizing likely increase in the index, but what is the index? Is it Y in training set or what? I initially thought that it is Y, but then we cannot tackle test set data, because there is no Y in it, as we have to predict it.**

**Answer**
*Adilet Gaparov*

	Yes. it is Y. you predict Y for test set and then based on your predictions, you do allocation


**Question**
*MouadB*

**@Aleksandr Finagin I start thinking about the optimization part the allocated enery (50K) have to be distributed over the test only or test+train . thanks**

**Answer**
*Vardan Ghazaryan*

	only across the test.


**Question**
*Mohanned Ahmed*

**Hi guys, how to get the extra energy? Mentioned in the formula: Index = extra energy * (Potential for increase in the Index **2) / 1000. Is it the same as pred_opt? (edited)**

**Answer**
*SOUAMES Annis*

	pred opt is the same as extra energy, it's the one you need to optimize with respect to the constraints in order to maximize the likely increase in index


**Question**
*Jonathan Whitaker*

**Can I ask for clarification on the formula:
Likely increase in the Index = extra energy * Potential for increase in the Index **2 / 1000
Is it :
Likely increase in the Index = extra energy * Potential for increase in the Index **(2 / 1000)
Likely increase in the Index = extra energy * (Potential for increase in the Index **2) / 1000**

**Answer**
*Devrim*

	Should be the second definitely, but indeed that would have been written in the way (2)


**Question**
*Paul Ashraf*

**Hey everyone I have a small question.  When it says we have to allocate 50,000 energy units across all galaxies. Does this mean we have to allocate the 50,00 across the 890 test cases or for each year we will allocate 50,000 across the galaxies that have a sample at this year?**

**Answer**
*Hessa Fahad*

	I understand it as  we should allocate it across the 890 test cases.  the 'galactic year' is not a unique feature. the main goal is the galaxy, 'galactic year' is one of its characteristic.

*Paul Ashraf*

	but there is duplicates in the galaxies names

*Hessa Fahad*

	yeah I've notice it. but for me I almost refer to the galaxies by index number so I cut the confusion for now at least. cuz they may give the same name for new born galaxy ?  like galaxy 'Large Magellanic Cloud (LMC)' has 19 'galactic year' ? so it's mostly the data scientist call to build his/her way to deal with it

*Paul Ashraf*

	okay will do the same


### Keywords: dataset, features, training

**Question**
*Mahmoud Bahaa*

**Hello guys, so some population values are actually in negative and the sum of 'Population, ages 15–64 (millions)', 'Population, ages 65 and older (millions)', 'Population, under age 5 (millions)' are not equal (or even close) to 'Population, total (millions)',  ?**

**Answer**
*solemn_leader*

	there might be aliens whose age is between 5 and 15 


**Question**
*A.I.*

**@Aleksandr Finagin the last four lines of the test set seem to be empty - is that by design?**

**Answer**
*Aleksandr Finagin*

	@A.I. It’s on purpose;)
*A.I.*

		Thanks. Then how do we treat them for the existence expectancy index - would they be "low" or not? (edited) 


**Question**
*Adilet Gaparov*

**Interesting enough: if we multiply "Population, total (millions)" by "Gross galactic product (GGP) per capita", we will not get "Gross galactic product (GGP), total"...   @Aleksandr Finagin**

**Answer**
*AG*

	Yeah, also renewable energy consumption (% of total) often often exceeds 100% :slightly_smiling_face:

*Aleksandr Finagin*

	@Adilet Gaparov @AG some data noise was included, but not a big one so that features are absolutely valid 


### Keywords: team, submission

**Question**
*Alibek Kaliyev*

**Hello! Can we somehow add a new member to the team if we already made a submission?:tired_face:**

**Answer**
*team_05_01*

	being in separate teams is better anyway! to each have 20 submissions :wink:


**Question**
*team_05_01*

**Can we know if the opt_pred has to be discrete or continuous?**

**Answer**
*Aleksandr Finagin*

	Continuous


**Question**
*Timur Abdualimov*

**Maybe you remove the number of restrictions on the number of total submissions and do it as on the kaggle, limit the number of submissions per day. That would be logical.**

**Answer**
*Vardan Ghazaryan*

	Allocation task would become useless in that case. People would simply brute force the optimal allocation


**Question**
*Oleg Bartov*

**I guess organisers should change task or data because there are a lot of teams who use this bug to fit on the results. For example "team_05" from Morocco. They have uploaded more than 20 submissions but their quantity is still 1**

**Answer**
*Kabir Abdulmajeed*

	@Aleksandr Finagin I want to believe that once this bug is fixed, every participant's score and submissions would be updated - there should be records, and those with 20 submissions would no longer be able to submit.


### Interesting Stuff

**Entry**
*Jonathan Whitaker*
*[Link to thread](https://prohackworkspace.slack.com/archives/C011W8E2UAX/p1589958685255400)*

**Looking at the years sequentially, you can get a decent score just by fitting the curve over time. Score of 0.79 (14th place) using no features besides the year, doing a separate model (GAM with a basic spline) for each galaxy.**

**Thread**
*team_05_01*

	hi, do you mean 0.079 ?

*Jonathan Whitaker*

	Yes, sorry 0.079. And @Zip that plot is just for a single galaxy. Each galaxy has data for multiple years. And then the test set covers some years for some galaxies. The idea here is that by simply looking at each galaxy individually and filling in the gaps with a simple model you can do decently without looking at the other features. Obviously not a winning approach but interesting nonetheless. Here's a notebook with the approach (not written for sharing but if it's of interest why not :slightly_smiling_face: ) https://colab.research.google.com/drive/1B8PqaSu8mF6w94PfTSRmvUqluwuB9Clm?usp=sharing


**Entry**
*Jonathan Whitaker*

**OK, I've made a super quick starter notebook. Open to questions and suggested improvements. I'll keep adding to it as I work on the problem :slightly_smiling_face:
https://colab.research.google.com/drive/1iiAqSu3sZxET5X6JqerYlmnURla0YQsB?usp=sharing**
