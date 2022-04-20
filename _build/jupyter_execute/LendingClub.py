#!/usr/bin/env python
# coding: utf-8

# # Pratical Hands-on skills
# 
# <figure><p>
# <img src="https://raw.githubusercontent.com/dataprofessor/infographic/master/01-Building-the-Machine-Learning-Model.JPG" alt="mlbuilding" width="80%" height="80%"></p><figcaption>Image Source From https://github.com/dataprofessor/infographic</figcaption></figure>
# 
# ## References
# 
# - https://www.lendingclub.com/investing/investor-education/interest-rates-and-fees
# - https://www.kaggle.com/code/faressayah/lending-club-loan-defaulters-prediction
# - https://www.kaggle.com/datasets/wordsforthewise/lending-club
# - [Peer-to-peer sturcture](https://brunch.co.kr/@beyondplatform/4)
# 
# ---

# # Lending Club
# 
# LendingClub is a peer-to-peer lending company headquartered in San Francisco California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC) and to offer loan trading on a secondary market. At its height LendingClub was the world's largest peer-to-peer lending platform. {cite:p}`wiki:LendingClub`
# 
# ```{figure} ./figs/lendingclub.png
# ---
# height: 250px
# name: directive-fig
# ---
# lending club website
# ```
# 
# ## About Personal loan
# 
# A personal loan is a loan which can be taken to meet unspecified financial needs. Today personal loan segment has diverted into many specialised loans. It can be taken for various purpose such as a wedding, traveling, paying education fee, medical emergencies or any undefined reason etc. The interest paid on a personal loan is in most cases higher than that payable on secured loans. {cite:p}`wiki:Unsecured_debt`
# 
# ## Business Understanding
# 
# Modified some of contents from {cite:p}`kaggle:aayush7kumar`.
# 
# The LendingClub company lends various types of loans to their customers. The company will make a decision for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision:
# 
# - If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company
# - If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company.
# 
# The data given contains the information about past loan applicants and whether they 'defaulted' or not. The aim is to identify patterns which indicate if a person is **likely** to default, which may be used for takin actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc.
# 
# When a person applies for a loan, there are two types of decisions that could be taken by the company:
# 
# 1. Loan accepted: If the company approves the loan, there are 3 possible scenarios described below:
#     - Fully paid: Applicant has fully paid the loan (the principal and the interest rate)
#     - Current: Applicant is in the process of paying the instalments, i.e. the tenure of the loan is not yet completed. These candidates are not labelled as 'defaulted'.
#     - Charged-off: Applicant has not paid the instalments in due time for a long period of time, i.e. he/she has defaulted on the loan
# 2. Loan rejected: The company had rejected the loan (because the candidate does not meet their requirements etc.). Since the loan was rejected, there is no transactional history of those applicants with the company and so this data is not available with the company (and thus in this dataset)
# 
# ## Business Metric
# 
# TBD
# 
# - It might be some explanation on why some of customer fails to pay the loan for reject purpose

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hvplot.pandas
from pathlib import Path

main_path = Path().absolute().parent
data_path = main_path.parent / 'data' / 'p2p' / 'lending_club' / 'processed'


# ## Exploratory Data Analysis

# ### Data Description
# 
# |LoanStatNew|Description|
# |---|---|
# |loan_amnt|The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.|
# |term|The number of payments on the loan. Values are in months and can be either 36 or 60.|
# |loan_status|Current status of the loan.|
# |int_rate|Interest Rate on the loan.|
# |installment|The monthly payment owed by the borrower if the loan originates.|
# |grade|LC assigned loan grade.|
# |sub_grade|LC assigned loan subgrade.|
# |emp_title|The job title supplied by the Borrower when applying for the loan.|
# |emp_length|Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. |
# |home_ownership|The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER
# |annual_inc|The self-reported annual income provided by the borrower during registration.|
# |verification_status|Indicates if income was verified by LC, not verified, or if the income source was verified.|
# |issue_d|The month which the loan was funded.|
# |purpose|A category provided by the borrower for the loan request. |
# |title|The loan title provided by the borrower.|
# |dti|A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrowe's self-reported monthly income.|
# |earliest_cr_line|The month the borrower's earliest reported credit line was opened.|
# |open_acc|The number of open credit lines in the borrower's credit file.|
# |pub_rec|Number of derogatory public records.|
# |pub_rec_bankruptcies|Number of public record bankruptcies.|
# |revol_bal|Total credit revolving balance.|
# |revol_util|"Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.|
# |total_acc|The total number of credit lines currently in the borrower's credit file.|
# |initial_list_status|The initial listing status of the loan. Possible values are – W, F|
# |application_type|Indicates whether the loan is an individual application or a joint application with two co-borrowers.|
# |mort_acc|Number of mortgage accounts.|
# |addr_state|The state provided by the borrower in the loan application.|

# In[2]:


# import data
df = pd.read_csv( data_path / 'accepted.csv')
df.info()


# ### loan status
# 
# Current status of the loan.

# In[3]:


df['loan_status'].value_counts().hvplot.bar(
    title='Loan Status Counts', xlabel='Loan Status', ylabel='Count', 
    width=500, height=350, yformatter='%d'
)


# ### loan_amnt & installment
# 
# - loan_amnt: The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
# - installment: The monthly payment owed by the borrower if the loan originates.

# In[4]:


loan_amnt = df.hvplot.hist(
    y='loan_amnt', by='loan_status', subplots=False, 
    width=400, height=400, bins=50, alpha=0.4, title='Loan Amount', 
    xlabel='Loan Amount', ylabel='Counts', legend='top',
    yformatter='%d'
)
loan_amnt


# In[5]:


installment = df.hvplot.hist(
    y='installment', by='loan_status', subplots=False, 
    width=400, height=400, bins=50, alpha=0.4, title='Installment', 
    xlabel='Installment', ylabel='Counts', legend='top',
    yformatter='%d'
)
installment


# In[6]:


installment_box = df.hvplot.box(
    y='installment', subplots=True, by='loan_status', width=250, height=400, 
    title='Installment', xlabel='Loan Status', ylabel='Installment', legend=False
)

loan_amnt_box = df.hvplot.box(
    y='loan_amnt', subplots=True, by='loan_status', width=250, height=400, 
    title='Loan Amount', xlabel='Loan Status', ylabel='Loan Amount', legend=False
)

loan_amnt_box + installment_box


# ### term & int_rate
# 
# - term: The number of payments on the loan. Values are in months and can be either 36 or 60.
# - int_rate: Interest Rate on the loan.

# In[7]:


term = df.groupby(['loan_status'])[['term']].value_counts().rename('Count').hvplot.bar()
term.opts(
    title="Term", xlabel='Term / Loan Status', ylabel='Count',
    width=500, height=450, show_legend=True, yformatter='%d'
)


# In[8]:


int_rate = df.hvplot.hist(
    y='int_rate', by='loan_status', subplots=False, 
    width=400, height=400, bins=30, alpha=0.4, 
    title='Interest Rate', 
    xlabel='Interest Rate', ylabel='Counts', legend='top',
    yformatter='%d'
)
int_rate


# usually sort-term has lower interest rate

# In[9]:


df.loc[:, ['loan_status', 'term', 'int_rate']].hvplot.hist(
    y='int_rate', groupby='loan_status', by='term', subplots=False, 
    width=400, height=400, bins=30, alpha=0.4, 
    title='Interest Rate by term', xlabel='Interest Rate', ylabel='Counts', legend='top',
    yformatter='%d', dynamic=False
)


# Interesting relation between interest rate and installment it that can be calculate by the following formula if using "Equal repayment of principal and interest"

# In[10]:


def cal_amount_erpi(loan_amnt, int_rate, term):
    """
    loan_anmt: loan amount
    int_rate: interest rate, percentage
    term: in month
    """
    int_rate_monthly = int_rate / 100 / 12
    payment_monthly = loan_amnt * int_rate_monthly
    total_to_pay = payment_monthly * (1 + int_rate_monthly)**term
    return total_to_pay / ((1 + int_rate_monthly)**term - 1)


# In[11]:


df_temp = df.loc[:, ['installment', 'loan_amnt', 'int_rate', 'term']].copy()
df_temp['term'] = df_temp['term'].str.strip('months').str.strip().astype(np.int32)
df_temp['installment_cal'] = df_temp.apply(lambda x: cal_amount_erpi(x['loan_amnt'], x['int_rate'], x['term']), axis=1)


# not all the payment are following "Equal repayment of principal and interest"

# In[12]:


df_temp_diff = (df_temp['installment_cal'] - df_temp['installment'])
df_diff = df_temp_diff.agg(['mean', 'std'])
print(f'Difference of Mean: {df_diff["mean"]:.4f} Standard Deviation {df_diff["std"]:.4f}')
df_temp_diff.loc[abs(df_temp_diff) > 1].hvplot.hist(
    subplots=False, width=400, height=400, bins=50, alpha=0.4, 
    title='Differences between calculated installment (Diff > 1)', xlabel='Diff', ylabel='Counts', legend='top',
    yformatter='%d',
)


# ### grade & sub_grade
# 
# - grade: LC assigned loan grade.
# - sub_grade: LC assigned loan subgrade.

# In[13]:


grade = df.groupby(['loan_status'])[['grade']].value_counts().sort_index().hvplot.bar(
    width=400, height=400, title='Grade Distribution', xlabel='Grade', ylabel='Count', 
    legend='top', yformatter='%d'
)
grade


# In[14]:


sub_grade = df.groupby(['loan_status'])[['sub_grade']].value_counts().sort_index().hvplot.barh(
    width=400, height=800, title='Sub-Grade Distribution', xlabel='Sub-Grade', ylabel='Count', 
    legend='top', xformatter='%d'
)
sub_grade


# usually people don't charge off at grade A - B, usually C grade are more often charge off. but cannot say that the people who has lower grade pay less on their loan.

# In[15]:


df.loc[df['grade'].isin(['E', 'F', 'G'])].groupby(['loan_status', 'grade'])[['grade']].value_counts().rename('count')    .hvplot.table(title='Grade Count in E-G')


# ### home_ownership & purpose
# 
# - home_ownership: The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER
# - purpose: A category provided by the borrower for the loan request.

# In[16]:


df.groupby(['loan_status'])[['home_ownership']].value_counts().rename('count').hvplot.table()


# In[17]:


home_ownership = df.groupby(['loan_status'])[['home_ownership']].value_counts().rename('Count').hvplot.bar()
home_ownership.opts(
    title='Home Ownership', xlabel='Home Ownership / Loan Status', ylabel='Count',
    width=700, height=450, show_legend=True, yformatter='%d'
)


# In[18]:


purpose = df.groupby(['loan_status'])[['purpose']].value_counts().rename('Count').hvplot.bar()
purpose.opts(
    title="Purpose", xlabel='Purpose / Loan Status', ylabel='Count',
    width=700, height=450, show_legend=True, yformatter='%d', xrotation=90
)


# ### annual_inc & verification_status
# 
# - annual_inc: The self-reported annual income provided by the borrower during registration.
# - verification_status: Indicates if income was verified by LC, not verified, or if the income source was verified

# In[19]:


df.groupby(['loan_status', 'verification_status'])['annual_inc'].describe().round(2).hvplot.table(
    title='Annual Income Table Description By Loan Status & Verification', height=200, width=700)


# In[20]:


(df.groupby(['loan_status'])[['verification_status']].value_counts() / df.groupby(['loan_status'])['verification_status'].count())    .rename('percentage').hvplot.table(title='Income Verified Rate', height=200)


# In[21]:


def is_outlier(x): 
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    upper = np.percentile(x, 75) + (iqr * 1.5)
    lower = np.percentile(x, 25) - (iqr * 1.5)

    return (x > upper) | (x < lower)

annual_inc = df.loc[~df.groupby(['loan_status', 'verification_status'])['annual_inc'].apply(is_outlier), 
    ['loan_status', 'verification_status', 'annual_inc']].hvplot.hist(
    y='annual_inc', by='loan_status', groupby='verification_status', subplots=False, 
    width=700, height=400, bins=40, alpha=0.4, title='Annual Income(1Q~3Q +/- 1.5*IQR) Distsribution', 
    xlabel='Annual Income', ylabel='Counts', legend='top', yformatter='%d', xformatter='%d', dynamic=False
)
annual_inc


# * Q: What is the difference between verified and not verified who has much high/lower income?

# In[22]:


df.loc[df.groupby(['loan_status', 'verification_status'])['annual_inc'].apply(is_outlier), 
['loan_status', 'verification_status', 'annual_inc']].groupby(['loan_status', 'verification_status'])['annual_inc'].describe()


# ### emp_title & emp_length
# 
# - emp_title: The job title supplied by the Borrower when applying for the loan.
# - emp_length: Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.

# In[23]:


check_null = lambda x: x.isnull().sum()
df_emp_null = df.loc[:, ['emp_title', 'emp_length']].apply([check_null, pd.Series.nunique]).rename(index={'<lambda>': 'nnull'}).reset_index().hvplot.table(
    title='Job Title & Employment length: NA values', height=100
)
df_emp_top20 = df['emp_title'].value_counts().reset_index().rename(columns={'index': 'emp_title', 'emp_title': 'count'})[:20].hvplot.table(
    title='Job Title Top 20'
)


# In[24]:


df_emp_null


# In[25]:


df_emp_top20


# In[26]:


df['emp_length'].fillna('unknown', inplace=True)
df['emp_title'].fillna('unknown', inplace=True)
df['emp_title'] = df['emp_title'].str.lower()  # Unify into lower cases


# In[27]:


df_emp_top20 = df['emp_title'].value_counts().reset_index().rename(columns={'index': 'emp_title', 'emp_title': 'count'})[:20].hvplot.table(
    title='Job Title Top 20'
)
df_emp_top20


# In[28]:


df_emp_bottom20 = df['emp_title'].value_counts().reset_index().rename(columns={'index': 'emp_title', 'emp_title': 'count'})[-20:].hvplot.table(
    title='Job Title Bottom 20'
)
df_emp_bottom20


# In[29]:


print(df['emp_title'].nunique())


# titles are not normalized(or structured), too many unique titles in the data.

# In[30]:


from itertools import product

loan_status_order = ['Charged Off', 'Fully Paid']
emp_length_order = ['unknown', '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
emp_length = df.groupby(['loan_status'])[['emp_length']].value_counts().reindex(list(product(*[loan_status_order, emp_length_order])))    .rename('Count').hvplot.barh(stacked=True, legend='right')
emp_length.opts(
    title='Employment Length in years', height=400, width=700, xlabel='Counts', ylabel='Employment Length in years', xformatter='%d'
)


# ### issue_d & earliest_cr_line
# 
# - issue_d: The month which the loan was funded.
# - earliest_cr_line: The month the borrower's earliest reported credit line was opened.
# 
# Red is the people who charge-off and the blue is the people who fully paied. Most people try to do the loan near the 2016 and started to create their credit line at 2000.

# In[31]:


import calendar

month_dict = {m: n for n, m in enumerate([calendar.month_abbr[i] for i in range(1, 13)], 1)}

df['issue_d'] = pd.to_datetime(df['issue_d'].str.split('-').apply(lambda x: f'{x[1]}-{month_dict.get(x[0])}'))
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'].str.split('-').apply(lambda x: f'{x[1]}-{month_dict.get(x[0])}'))


# In[32]:


fully_paid = df.loc[df['loan_status']=='Fully Paid', 'issue_d'].hvplot.hist(bins=35) 
charged_off = df.loc[df['loan_status']=='Charged Off', 'issue_d'].hvplot.hist(bins=35)

# fully_paid * charged_off
loan_issue_date = (fully_paid * charged_off).opts(
    title='Loan Issue Date Distribution', xlabel='Loan Issue Date', ylabel='Count',
    width=350, height=350, legend_cols=2, legend_position='top_right'
).opts(xrotation=45, yformatter='%d')

fully_paid = df.loc[df['loan_status']=='Fully Paid', 'earliest_cr_line'].hvplot.hist(bins=35) 
charged_off = df.loc[df['loan_status']=='Charged Off', 'earliest_cr_line'].hvplot.hist(bins=35)

earliest_cr_line = (fully_paid * charged_off).opts(
    title='Earliest reported credit line', xlabel='earliest_cr_line', ylabel='Count',
    width=350, height=350, legend_cols=2, legend_position='top_right'
).opts(xrotation=45, yformatter='%d')

loan_issue_date + earliest_cr_line


# * Q: Are there anyone who applied before the credit line is reported?

# In[33]:


issue_report = df['issue_d'] < df['earliest_cr_line']
print(f'The percentage that who applied before the credit line is reported: {(issue_report).sum() / len(df)}')


# * Q: Are there any difference between months?

# In[34]:


df['issue_d_month'] = df['issue_d'].dt.month

issue_d_month = df.groupby(['loan_status'])[['issue_d_month']].value_counts().rename('Count').hvplot.bar()
issue_d_month.opts(
    title="Issue Date Distribution in every month by Loan Status", xlabel='Month', ylabel='Count',
    width=700, height=450, show_legend=True, yformatter='%d'
)


# ### title
# 
# title is duplicated with the `purpose` column, will drop it later

# In[35]:


print(df['title'].isnull().sum())


# In[36]:


df['title'] = df['title'].str.lower()
df['title'].value_counts()[:10]


# ### dti, open_acc, total_acc
# 
# ```{admonition} What Is Debt-to-Income Ratio?
# 
# Your debt-to-income ratio compares your debt payments to your monthly gross income, or how much you earn each month before taxes and other deductions. Your DTI ratio gives lenders a clearer picture of your current debt and income, and is used to determine how much money you can afford to responsibly borrow.
# 
# Monthly debt may include:
# 
# - Minimum credit card payments
# - Loan payments (such as car payments, student loan payments, personal loans, and other loan payments)
# - Monthly alimony or child support payments
# - Rent payment or mortgage payments
# - Other debts included in your credit report
# 
# ```
# 
# - dti: A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower's self-reported monthly income.
# - open_acc: The number of open credit lines in the borrower's credit file.
# - total_acc: The total number of credit lines currently in the borrower's credit file.
# 
# [NerdWallet website](https://www.nerdwallet.com/reviews/loans/personal-loans/lendingclub-personal-loans) says that the maximum allowed DTI ratio is 40% for single applicants and 35% for joint applicants. In the [Lending Club website](https://www.lendingclub.com/loans/resource-center/calculating-debt-to-income), seems like over 40% DTI is not a good signal, they suggest some way to improve the DTI ratio.

# In[37]:


df['dti'].describe().reset_index().hvplot.table(title='DTI Table Description', height=250)
# Can DTI be 999?


# It seems like the DTI over 60 can be treated as outlier data, may need to drop them.

# In[38]:


df.loc[df['dti'] > 40].groupby(['loan_status'])['dti'].describe().hvplot.table(title='DTI > 40% Table Description', height=100)


# In[39]:


dti_sub = df.loc[df['dti'] <= 40].hvplot.hist(
    y='dti', by='loan_status', bins=50, width=400, height=350, subplots=False, 
    title="dti(<=50) Distribution", xlabel='dti', ylabel='Count', shared_axes=False,
    alpha=0.4, legend='top', yformatter='%d'
)

dti_sub2 = df.loc[df['dti'] > 40].hvplot.hist(
    y='dti', by='loan_status', bins=100, width=400, height=350, subplots=False, 
    title="dti(>50) Distribution", xlabel='dti', ylabel='Count', shared_axes=False,
    alpha=0.4, legend='top', yformatter='%d'
)

dti_sub + dti_sub2


# In[40]:


open_acc = df.hvplot.hist(
    y='open_acc', by='loan_status', bins=50, width=450, height=350, 
    title='The number of open credit lines', xlabel='The number of open credit lines', ylabel='Count', 
    alpha=0.4, legend='top', yformatter='%d'
)

total_acc = df.hvplot.hist(
    y='total_acc', by='loan_status', bins=50, width=450, height=350, 
    title='The total number of credit lines', xlabel='The total number of credit lines', ylabel='Count', 
    alpha=0.4, legend='top', yformatter='%d'
)

open_acc + total_acc


# ### revol_bal & revol_util
# 
# ```{admonition} What is revolving balance?
# 
# In credit card terms, a revolving balance is the portion of credit card spending that goes unpaid at the end of a billing cycle. 
# ```
# 
# https://www.capitalone.com/learn-grow/money-management/revolving-credit-balance/
# 
# - revol_bal: Total credit revolving balance.
# - revol_util: Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.

# In[41]:


df.groupby(['loan_status'])['revol_bal'].describe().round(2).reset_index().hvplot.table(title='Revolving Balance Table Description', height=100)


# In[42]:


revol_bal = df.hvplot.hist(
    y='revol_bal', by='loan_status', bins=50, width=350, height=400, 
    title='Revolving Balance', xlabel='Revolving balance', ylabel='Count', 
    alpha=0.4, legend='top', yformatter='%d', xformatter='%d'
).opts(xrotation=45)

revol_bal_sub = df.loc[df['revol_bal']<=250000].hvplot.hist(
    y='revol_bal', by='loan_status', bins=50, width=350, height=400, 
    title='Revolving Balance(<=250000)', xlabel='Revolving balance', ylabel='Count', 
    alpha=0.4, legend='top', yformatter='%d', xformatter='%d', shared_axes=False
)
revol_bal + revol_bal_sub


# In[43]:


revol_util = df.hvplot.hist(
    y='revol_util', by='loan_status', bins=50, width=350, height=400, 
    title='Revolving line utilization rate', xlabel='Revolving line utilization rate', ylabel='Count', 
    alpha=0.4, legend='top'
).opts(yformatter='%d')

revol_util_sub = df[df['revol_util'] < 120].hvplot.hist(
    y='revol_util', by='loan_status', bins=50, width=350, height=400, 
    title='Revolving line utilization rate (< 120)', xlabel='Revolving line utilization rate', ylabel='Count', 
    shared_axes=False, alpha=0.4, legend='top'
).opts(yformatter='%d')

revol_util + revol_util_sub


# ### pub_rec, pub_rec_bankruptcies & mort_acc
# 
# 
# ```{admonition} What is derogatory record?
# A derogatory public record is negative information on your credit report that is of a more serious nature and has become a matter of public record. It usually consists of bankruptcy filings, civil court judgments, foreclosures and tax liens. In some states, child support delinquencies are also a matter of public record.
# ```
# 
# https://budgeting.thenest.com/derogatory-public-record-mean-25266.html
# 
# 
# ```{admonition} What is a mortgage?
# The mortgage refers to a loan used to purchase or maintain a home, land, or other types of real estate. Usually paying the mortgage consistently will increse the credit score.
# ```
# 
# https://www.investopedia.com/terms/m/mortgage.asp
# https://www.investopedia.com/articles/personal-finance/031215/how-mortgages-affect-credit-scores.asp
# 
# - pub_rec: Number of derogatory public records.
# - pub_rec_bankruptcies: Number of public record bankruptcies.
# - mort_acc: Number of mortgage accounts.
# 
# From the data we can process these data as binary who had never have a public record versus more than once.

# In[44]:


pub_rec = df.groupby(['loan_status'])['pub_rec'].value_counts().rename('Count').hvplot.barh(
    title='The number of derogatory', xlabel='The number of derogatory', ylabel='Count',
    width=400, height=800, xformatter='%d'
)
pub_rec


# In[45]:


pub_rec_bankruptcies = df.groupby(['loan_status'])['pub_rec_bankruptcies'].value_counts().rename('Count').hvplot.barh(
    title='The number of public record bankruptcies', xlabel='The number of public record bankruptcies', ylabel='Count',
    width=400, height=600, xformatter='%d'
)
pub_rec_bankruptcies


# In[46]:


df['mort_acc'].describe().round(2)


# In[47]:


mort_acc = df.groupby(['loan_status'])['mort_acc'].value_counts().rename('Count').hvplot.barh(
    title='The number of mortgage accounts', xlabel='The number of mortgage accounts', ylabel='Count',
    width=400, height=700, xformatter='%d'
)

print(df['mort_acc'].isnull().sum())

mort_acc


# ### initial_list_status, application_type & addr_state
# 
# - initial_list_status: The initial listing status of the loan. Possible values are – W, F
# - application_type: Indicates whether the loan is an individual application or a joint application with two co-borrowers.
# - addr_state: The state provided by the borrower in the loan application.

# In[48]:


initial_list_status = df.groupby(['loan_status'])['initial_list_status'].value_counts().rename('Count').hvplot.bar(
    title='The initial listing status of the loan', xlabel='The initial listing status of the loan', ylabel='Count',
    width=400, height=400, yformatter='%d'
)
initial_list_status


# In[49]:


application_type = df.groupby(['loan_status'])['application_type'].value_counts().rename('Count').hvplot.bar(
    title='The application type', xlabel='The application type', ylabel='Count',
    width=400, height=400, yformatter='%d'
)
application_type


# In[50]:


addr_state = df.groupby(['loan_status'])['addr_state'].value_counts().rename('Count').hvplot.barh(
    title='The state provided by the borrower', xlabel='The state', ylabel='Count',
    width=500, height=850, xformatter='%d', legend='right'
)
addr_state


# ## Data Preprocessing
# 
# - Drop columns
# - Missing values
# - Detecting outlieres

# In[51]:


# reload the data
df = pd.read_csv( data_path / 'accepted.csv')
print(f'Data shape: {df.shape}')
print(df.columns)


# According to our EDA, We will not use following columns: 
# 
# - `title`: duplicated with `purpose`
# - `emp_title`: too many unique jobs, but seems like some of them are duplicated
# - `issue_d`, `earliest_cr_line`: nothing interesting
# 
# Other columns 
# - `term`: change its' type to integer
# - `grade`, `sub_grade`, `home_ownership`, `purpose`, `initial_list_status`, `application_type`, `addr_state`: do label encoding
# - `emp_length`: add 'unknown' for NaN values and do the label encoding
# - `verification_status`: convert `source verified` as `verified` together and do the label encoding
# - `dti`: drop which dti over 60%
# - `revol_bal`: drop which has over $ 250,000 balance
# - `revol_util`: drop which has over 120% utilization rate
# - `pub_rec`, `pub_rec_bankruptcies`: convert as binary, who has ever had the record or not
# - `mort_acc`: can convert to categories [0, 1, 2, 3, 4, 5, 5+] and do the label encoding
# - `loan_status`: target column, do the label encoding

# In[52]:


from collections import defaultdict

# drop columns values
df.drop(columns=['title', 'emp_title', 'issue_d', 'earliest_cr_line'], inplace=True)
print(f"- Dropped columns: {['title', 'emp_title', 'issue_d', 'earliest_cr_line']}")

# term
df['term'] = df['term'].str.rstrip('months').astype(int)
print('- Type changed into integer: term')

# need a label encoding
encode_dict = defaultdict(dict)
for c in ['grade', 'sub_grade', 'home_ownership', 'purpose', 'initial_list_status', 'application_type', 'addr_state']:
    encode_dict[c]['v2i'] = {v: i for i, v in enumerate(sorted(df[c].unique()))}
    encode_dict[c]['i2v'] = {i: v for v, i in encode_dict[c]['v2i'].items()}
    df[c] = df[c].map(encode_dict[c]['v2i'])
    print(f'- Label encoded: {c}')

# emp_length
c = 'emp_length'
df[c].fillna('unknown', inplace=True)
emp_values = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
encode_dict[c]['v2i'] = {v: i for i, v in enumerate(emp_values)}
encode_dict[c]['v2i']['unknown'] = 99
encode_dict[c]['i2v'] = {i: v for v, i in encode_dict[c]['v2i'].items()}
df[c] = df[c].map(encode_dict[c]['v2i'])
print(f'- Label encoded: {c}')

# verification_status
c = 'verification_status'
df[c].replace(to_replace='Source Verified', value='Verified', inplace=True)
print(f'- Merged "Source Verified" into "Verified": {c}')
veri_values = ['Not Verified', 'Verified']
encode_dict[c]['v2i'] = {v: i for i, v in enumerate(veri_values)}
encode_dict[c]['i2v'] = {i: v for v, i in encode_dict[c]['v2i'].items()}
df[c] = df[c].map(encode_dict[c]['v2i'])
print(f'- Label encoded: {c}')

# dti, revol_bal, revol_util
for c, thres in zip(['dti', 'revol_bal', 'revol_util'], [60, 250000, 120]):
    drop_idx = df.index[df[c] > thres]
    df.drop(index=drop_idx, inplace=True)
    print(f'- Dropped # of {(len(drop_idx)/len(df))*100:.2f}% ({len(drop_idx)}) data: {c}')

# pub_rec, pub_rec_bankruptcies
for c in ['pub_rec', 'pub_rec_bankruptcies']:
    df[c].apply(lambda x: 0 if x == 0 else 1)
    print(f'- Convert into binary feature data: {c}')

# mort_acc
c = 'mort_acc'
df.loc[df[c] > 5, c] = 6
mort_values = ['0', '1', '2', '3', '4', '5', '5+']
encode_dict[c]['v2i'] = {v: i for i, v in enumerate(mort_values)}
encode_dict[c]['i2v'] = {i: v for v, i in encode_dict[c]['v2i'].items()}
df.loc[df[c] > 5, c] = 6
print(f'- Label encoded: {c}')

# target column encoding
c = 'loan_status'
encode_dict[c]['v2i'] = {'Fully Paid': 0, 'Charged Off': 1}
encode_dict[c]['i2v'] = {i: v for v, i in encode_dict[c]['v2i'].items()}
df[c] = df[c].map(encode_dict[c]['v2i'])
print(f'- Label encoded: {c}')

df.reset_index(drop=True, inplace=True)


# Check missing data

# In[53]:


for column in df.columns:
    missing_col = df[column].isnull().sum()
    if missing_col != 0:
        missing_percentage = (missing_col / len(df)) * 100
        print(f"'{column}': number of missing values {missing_col}({missing_percentage:.3f}%)")


# Since the data is not that much we will drop these records.

# In[54]:


for c in ['dti', 'revol_util', 'mort_acc', 'pub_rec_bankruptcies']:
    drop_idx = df.index[df[c].isnull()]
    df.drop(index=drop_idx, inplace=True)
df.reset_index(drop=True, inplace=True)
df_loan_status_counts = df['loan_status'].value_counts()
print(f'The total number of records is {len(df)}')
print(f"- loan_status = Fully Paid: {df_loan_status_counts.iloc[0]}")
print(f"- loan_status = Charged Off: {df_loan_status_counts.iloc[1]}")


# In[55]:


# Save the processed data
# import pickle
# with (data_path / 'encode_dict.pickle').open('wb') as file:
#     pickle.dump(encode_dict, file)
# df.to_csv(data_path / 'accepted_processed.csv', index=False, encoding='utf-8')


# ## Modeling

# In[56]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the processed data
df = pd.read_csv(data_path / 'accepted_processed.csv')
y = df['loan_status']
X = df.loc[:, ~df.columns.isin(['loan_status'])]


# In[57]:


spiliter = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
train_idx, test_idx = list(*spiliter.split(X, y))
X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_test, y_test = X.loc[test_idx], y.loc[test_idx]


# In[58]:


model = RandomForestClassifier(n_jobs=8, verbose=1)
model.fit(X_train, y_train)


# In[59]:


y_pred = model.predict(X_test)
rpt = classification_report(y_true=y_test, y_pred=y_pred)
print(rpt)


# ## Explanation on Models

# In[60]:


import shap
shap.initjs()


# In[61]:


np.random.seed(78)
# pick 20 right predictions for each class
y_correct = y_test[y_test == y_pred]
num_samples = 10
rnd_idx_0 = np.random.choice(y_correct[y_correct == 0].index, num_samples) 
rnd_idx_1 = np.random.choice(y_correct[y_correct == 1].index, num_samples)
rnd_idx = np.concatenate((rnd_idx_0, rnd_idx_1))
y_sampled = y_correct.loc[rnd_idx]
X_sampled = X_test.loc[rnd_idx]


# In[62]:


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sampled, y_sampled)


# In[63]:


shap.summary_plot(shap_values, X_sampled, alpha=0.8, color_bar=True)


# In[64]:


label = 1
shap.dependence_plot('sub_grade', shap_values[label], features=X_sampled)


# In[65]:


shap_values[1].shape


# In[66]:


label = 0
sample_index = 0
shap.force_plot(explainer.expected_value[label], shap_values[label][sample_index], features=X_sampled.iloc[sample_index])


# In[67]:


label = 1
sample_index = 10
shap.force_plot(explainer.expected_value[label], shap_values[label][sample_index], features=X_sampled.iloc[sample_index])

