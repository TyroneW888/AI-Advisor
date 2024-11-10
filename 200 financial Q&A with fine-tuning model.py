#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dictionary containing 200 financial questions and answers
financial_qna = {
    "What is a budget?": "A budget is a financial plan that outlines an individual’s or organization’s expected income and expenses over a specified period.",
    "Why is saving money important?": "Saving money provides financial security, helps achieve goals, and prepares you for emergencies.",
    "What is an emergency fund?": "An emergency fund is money set aside to cover unexpected expenses, such as medical bills or car repairs.",
    "How much should be in an emergency fund?": "Generally, it should cover 3-6 months’ worth of living expenses.",
    "What is a savings account?": "A savings account is a bank account that earns interest on your deposits and is used for storing money safely.",
    "What is interest?": "Interest is the cost of borrowing money or the reward for saving money, typically expressed as a percentage.",
    "What is compound interest?": "Compound interest is interest calculated on the initial principal and also on the accumulated interest from previous periods.",
    "What is a credit score?": "A credit score is a numerical representation of your creditworthiness based on your credit history.",
    "What affects your credit score?": "Factors include payment history, credit utilization, length of credit history, types of credit, and recent credit inquiries.",
    "Why is a good credit score important?": "A good credit score can help you qualify for better interest rates on loans and credit cards and improve your financial credibility.",
    "What is a checking account?": "A checking account is a bank account used for daily transactions such as deposits, withdrawals, and bill payments.",
    "What is the difference between a debit card and a credit card?": "A debit card uses money from your bank account, while a credit card allows you to borrow funds up to a certain limit.",
    "What is a loan?": "A loan is a sum of money borrowed that must be repaid, usually with interest.",
    "What is a mortgage?": "A mortgage is a loan used to buy real estate, with the property itself serving as collateral.",
    "What is a down payment?": "A down payment is an upfront sum paid for a large purchase, such as a house or car, typically expressed as a percentage of the total price.",
    "What is an asset?": "An asset is something valuable that you own, which can include cash, property, stocks, and bonds.",
    "What is a liability?": "A liability is a financial obligation or debt that an individual or business owes.",
    "What is net worth?": "Net worth is the total value of your assets minus your liabilities.",
    "What is diversification?": "Diversification is the practice of spreading your investments across different assets to reduce risk.",
    "What is an investment portfolio?": "An investment portfolio is a collection of assets such as stocks, bonds, and real estate owned by an individual or institution.",
    "What are stocks?": "Stocks represent shares of ownership in a company and entitle the holder to a portion of the company’s profits.",
    "What are bonds?": "Bonds are debt securities issued by governments or corporations to raise funds, with a promise to repay with interest.",
    "What is a dividend?": "A dividend is a portion of a company’s earnings distributed to shareholders.",
    "What is a mutual fund?": "A mutual fund is a pooled investment vehicle managed by professionals, which invests in a diversified portfolio of assets.",
    "What is an ETF (Exchange-Traded Fund)?": "An ETF is an investment fund traded on stock exchanges, holding a collection of assets like stocks or bonds.",
    "What is the stock market?": "The stock market is a marketplace where investors buy and sell shares of publicly traded companies.",
    "What is a bear market?": "A bear market is a period when stock prices are falling, typically by 20% or more from recent highs.",
    "What is a bull market?": "A bull market is a period when stock prices are rising, indicating investor confidence and economic growth.",
    "What is risk tolerance?": "Risk tolerance is the degree of variability in investment returns an investor is willing to withstand.",
    "What is a financial advisor?": "A financial advisor is a professional who helps individuals manage their finances, including investments, budgeting, and retirement planning.",
    "What is retirement planning?": "Retirement planning involves setting financial goals and creating a strategy to achieve a secure and comfortable retirement.",
    "What is a 401(k)?": "A 401(k) is an employer-sponsored retirement savings plan that allows employees to contribute pre-tax income.",
    "What is an IRA (Individual Retirement Account)?": "An IRA is a tax-advantaged account that individuals use to save for retirement independently.",
    "What is a Roth IRA?": "A Roth IRA is a type of retirement account where contributions are made with after-tax dollars, allowing for tax-free withdrawals in retirement.",
    "What is a pension?": "A pension is a retirement plan where an employer provides regular income to retired employees based on their earnings and years of service.",
    "What is social security?": "Social security is a government program that provides financial assistance to retirees and disabled individuals.",
    "What is inflation?": "Inflation is the rate at which the general level of prices for goods and services rises, reducing purchasing power.",
    "What is deflation?": "Deflation is the decrease in the general price level of goods and services, often leading to increased purchasing power.",
    "What is the Consumer Price Index (CPI)?": "CPI is an index that measures the average change in prices over time for a basket of consumer goods and services.",
    "What is a budget deficit?": "A budget deficit occurs when expenses exceed revenue, leading to a shortfall.",
    "What is a budget surplus?": "A budget surplus occurs when income exceeds expenses, resulting in excess funds.",
    "What is GDP (Gross Domestic Product)?": "GDP is the total monetary value of all goods and services produced within a country over a specified period.",
    "What is a recession?": "A recession is a period of economic decline marked by falling GDP, reduced spending, and rising unemployment.",
    "What is the difference between saving and investing?": "Saving is setting aside money for future use, typically with low risk, while investing involves putting money into assets that can grow in value but come with higher risk.",
    "What is financial literacy?": "Financial literacy is the ability to understand and effectively use financial skills, including budgeting, investing, and managing debt.",
    "What is an IPO (Initial Public Offering)?": "An IPO is the first time a company offers its stock for public purchase on the stock exchange.",
    "What is a stock split?": "A stock split occurs when a company divides its existing shares into multiple shares to boost liquidity.",
    "What is dollar-cost averaging?": "Dollar-cost averaging is an investment strategy that involves regularly investing a fixed amount regardless of market conditions.",
    "What is a balance sheet?": "A balance sheet is a financial statement that shows a company’s assets, liabilities, and equity at a specific point in time.",
    "What is an income statement?": "An income statement is a financial report that shows a company’s revenues, expenses, and profits over a period.",
    "What is cash flow?": "Cash flow is the movement of money in and out of a business or individual’s accounts.",
    "What is liquidity?": "Liquidity refers to how easily an asset can be converted into cash without affecting its price.",
    "What is a certificate of deposit (CD)?": "A CD is a savings product with a fixed interest rate and fixed maturity date, usually offering higher interest than savings accounts.",
    "What is a fixed expense?": "A fixed expense is a recurring cost that does not change from month to month, such as rent or a mortgage payment.",
    "What is a variable expense?": "A variable expense is a cost that can change from month to month, such as utilities or groceries.",
    "What is credit utilization?": "Credit utilization is the percentage of your total credit limit that you are using.",
    "What is a credit report?": "A credit report is a detailed record of your credit history used by lenders to assess your creditworthiness.",
    "What is APR (Annual Percentage Rate)?": "APR is the annual rate charged for borrowing or earned through an investment, expressed as a percentage.",
    "What is a FICO score?": "A FICO score is a type of credit score created by the Fair Isaac Corporation, used by lenders to assess credit risk.",
    "What is the difference between a secured and an unsecured loan?": "A secured loan is backed by collateral, while an unsecured loan is not.",
    "What is a payday loan?": "A payday loan is a short-term, high-interest loan intended to be repaid on the borrower’s next payday.",
    "What is bankruptcy?": "Bankruptcy is a legal process where individuals or businesses unable to meet their debt obligations seek relief from some or all of their debts.",
    "What is a line of credit?": "A line of credit is a flexible loan from a bank or financial institution that allows borrowing up to a certain limit.",
    "What is home equity?": "Home equity is the value of a homeowner’s interest in their property, calculated as the property’s market value minus any mortgages owed.",
    "What is a credit limit?": "A credit limit is the maximum amount you can borrow on a credit card or line of credit.",
    "What is a late fee?": "A late fee is a charge incurred when a payment is not made by its due date.",
    "What is an overdraft fee?": "An overdraft fee is a charge for withdrawing more money than what is available in your bank account.",
    "What is a personal loan?": "A personal loan is an unsecured loan that can be used for various personal expenses.",
    "What is a co-signer?": "A co-signer is someone who agrees to take responsibility for a loan if the primary borrower defaults.",
    "What is a mortgage broker?": "A mortgage broker is a professional who helps individuals find and secure a mortgage loan from lenders.",
    "What is private mortgage insurance (PMI)?": "PMI is insurance that protects the lender in case the borrower defaults on a mortgage.",
    "What is an adjustable-rate mortgage (ARM)?": "An ARM is a mortgage with an interest rate that can change periodically based on market conditions.",
    "What is a fixed-rate mortgage?": "A fixed-rate mortgage has an interest rate that remains the same throughout the loan term.",
    "What is amortization?": "Amortization is the process of spreading out a loan into a series of fixed payments over time.",
    "What is refinancing?": "Refinancing is the process of replacing an existing loan with a new loan, usually with better terms.",
    "What is a HELOC (Home Equity Line of Credit)?": "A HELOC is a line of credit secured by the equity in your home.",
    "What is debt consolidation?": "Debt consolidation is the process of combining multiple debts into a single loan with a lower interest rate.",
    "What is an annuity?": "An annuity is a financial product that provides regular payments, typically for retirement income.",
    "What is a capital gain?": "A capital gain is the profit earned from selling an asset for more than its purchase price.",
    "What is a capital loss?": "A capital loss occurs when an asset is sold for less than its purchase price.",
    "What is a financial statement?": "A financial statement is a document that provides an overview of an individual’s or company’s financial condition.",
    "What is an audit?": "An audit is an examination of financial records to ensure accuracy and compliance with regulations.",
    "What is a credit union?": "A credit union is a member-owned financial institution that offers similar services to a bank, often with better rates.",
    "What is the Federal Reserve?": "The Federal Reserve is the central bank of the United States, responsible for monetary policy.",
    "What is a money market account?": "A money market account is a type of savings account that typically offers higher interest rates and limited check-writing ability.",
    "What is an expense ratio?": "An expense ratio is the fee that mutual funds or ETFs charge investors for managing their money.",
    "What is rebalancing a portfolio?": "Rebalancing involves adjusting the weightings of assets in an investment portfolio to maintain the desired level of risk.",
    "What is an index fund?": "An index fund is a type of mutual fund designed to match or track the performance of a specific market index.",
    "What is an S&P 500?": "The S&P 500 is a stock market index that tracks the performance of 500 large U.S. companies.",
    "What is a credit freeze?": "A credit freeze prevents creditors from accessing your credit report, helping protect against identity theft.",
    "What is a 529 plan?": "A 529 plan is a tax-advantaged savings plan designed to encourage saving for future education expenses.",
    "What is a tax deduction?": "A tax deduction reduces the amount of income subject to tax, potentially lowering your tax bill.",
    "What is a tax credit?": "A tax credit directly reduces the amount of tax you owe, often more valuable than a deduction.",
    "What is a W-2 form?": "A W-2 form reports an employee’s annual wages and the amount of taxes withheld from their paycheck.",
    "What is a 1099 form?": "A 1099 form reports various types of income other than wages, salaries, and tips.",
    "What is estate planning?": "Estate planning is the process of arranging for the management and distribution of an individual’s assets after death.",
    "What is a will?": "A will is a legal document that outlines how a person’s assets should be distributed after their death.",
    "What is power of attorney?": "Power of attorney is a legal authorization for one person to act on another’s behalf in financial or legal matters.",
    "What is a trust?": "A trust is a fiduciary arrangement where a trustee holds and manages assets for the benefit of beneficiaries.",
    "What is probate?": "Probate is the legal process of administering a deceased person’s estate and distributing assets to beneficiaries.",
    "What is life insurance?": "Life insurance is a contract where the insurer pays a beneficiary a sum of money upon the insured person’s death.",
    "What is term life insurance?": "Term life insurance provides coverage for a specified period, paying a benefit if the insured dies during the term.",
    "What is whole life insurance?": "Whole life insurance is a type of permanent life insurance that provides coverage for the insured's lifetime and includes a cash value component.",
    "What is an insurance premium?": "An insurance premium is the amount paid for an insurance policy.",
    "What is financial independence?": "Financial independence is the state of having enough wealth and income to cover one’s living expenses without needing to work actively.",
    "What is a credit inquiry?": "A credit inquiry is a request made by a financial institution to check your credit report, which can be either hard or soft.",
    "What is a hard inquiry?": "A hard inquiry occurs when a lender checks your credit report as part of the decision-making process for a loan or credit card application and can affect your credit score.",
    "What is a soft inquiry?": "A soft inquiry occurs when a credit check is done as part of a background check or pre-approval process and does not affect your credit score.",
    "What is a financial goal?": "A financial goal is a specific, measurable target related to money, such as saving for retirement, buying a home, or paying off debt.",
    "What is debt-to-income ratio?": "The debt-to-income ratio is a percentage that compares your monthly debt payments to your gross monthly income.",
    "What is a tax bracket?": "A tax bracket is a range of income taxed at a particular rate under a progressive tax system.",
    "What is a progressive tax?": "A progressive tax is a tax system in which the rate increases as the taxable amount increases, meaning higher income earners pay a higher percentage.",
    "What is a regressive tax?": "A regressive tax is a tax system where the rate decreases as the taxable base increases, placing a higher burden on lower-income earners.",
    "What is a flat tax?": "A flat tax is a tax system with a constant tax rate, regardless of income level.",
    "What is tax deferral?": "Tax deferral allows individuals or companies to postpone tax liability to a future period, as seen in retirement accounts like 401(k)s.",
    "What is an estate tax?": "An estate tax is a tax imposed on the value of a deceased person’s estate before distribution to heirs.",
    "What is a gift tax?": "A gift tax is a federal tax on the transfer of money or property to another person while receiving nothing (or less than full value) in return.",
    "What is depreciation?": "Depreciation is the reduction in the value of an asset over time due to wear and tear or obsolescence.",
    "What is appreciation?": "Appreciation is an increase in the value of an asset over time.",
    "What is a financial planner?": "A financial planner is a professional who helps individuals create strategies for managing their financial future, including savings, investments, and retirement.",
    "What is a fiduciary?": "A fiduciary is a person or organization that acts on behalf of another, putting their client’s interests ahead of their own.",
    "What is a recession-proof investment?": "Recession-proof investments are assets that tend to retain or increase their value during economic downturns, such as utility stocks or consumer staples.",
    "What is an inflation hedge?": "An inflation hedge is an investment that protects against the diminishing value of a currency, such as real estate or commodities.",
    "What is a savings bond?": "A savings bond is a government bond that offers a fixed interest rate over a fixed period and is considered a low-risk investment.",
    "What is dollar depreciation?": "Dollar depreciation is a decline in the value of the U.S. dollar compared to other currencies, impacting the cost of imports and exports.",
    "What is capital preservation?": "Capital preservation is an investment strategy focused on preventing the loss of capital and maintaining the value of an investment over time.",
    "What is a hedge fund?": "A hedge fund is an investment fund that uses various strategies, including leverage and derivatives, to generate high returns for wealthy investors.",
    "What is leverage?": "Leverage is the use of borrowed money to increase the potential return of an investment.",
    "What is margin trading?": "Margin trading is the practice of borrowing funds from a broker to buy more securities than you could with your own funds alone.",
    "What is a stop-loss order?": "A stop-loss order is an instruction to sell a security when it reaches a certain price, used to limit an investor’s loss.",
    "What is asset allocation?": "Asset allocation is the process of spreading investments among different asset categories, such as stocks, bonds, and cash, to reduce risk.",
    "What is reinsurance?": "Reinsurance is insurance that an insurance company purchases to mitigate risk by sharing potential large losses with other insurers.",
    "What is an escrow account?": "An escrow account is a third-party account used to hold funds during a transaction, such as the purchase of real estate, until specific conditions are met.",
    "What is principal?": "Principal refers to the original sum of money invested or borrowed, excluding any interest or earnings.",
    "What is a high-yield savings account?": "A high-yield savings account offers a higher interest rate than a traditional savings account, helping your savings grow faster.",
    "What is a money market fund?": "A money market fund is a type of mutual fund that invests in short-term, low-risk securities and provides high liquidity.",
    "What is active investing?": "Active investing involves frequent buying and selling of investments to outperform the market index.",
    "What is passive investing?": "Passive investing aims to replicate the performance of a specific index or benchmark, typically involving less frequent trading.",
    "What is a yield curve?": "A yield curve is a graph that shows the relationship between interest rates and the maturity dates of debt securities.",
    "What is a credit default swap (CDS)?": "A CDS is a financial derivative that allows an investor to swap or offset their credit risk with that of another investor.",
    "What is a venture capital firm?": "A venture capital firm is an investment firm that provides funding to startups and small businesses with long-term growth potential in exchange for equity.",
    "What is crowdfunding?": "Crowdfunding is the practice of raising small amounts of money from a large number of people, typically via the internet, to fund a project or business.",
    "What is financial leverage?": "Financial leverage refers to the use of debt to acquire additional assets, increasing the potential return on investment.",
    "What is equity financing?": "Equity financing is the process of raising capital through the sale of shares in a company.",
    "What is debt financing?": "Debt financing is the process of raising capital by borrowing money that must be repaid, often with interest.",
    "What is portfolio rebalancing?": "Portfolio rebalancing is the process of realigning the weightings of a portfolio’s assets to maintain a desired risk level.",
    "What is a blue-chip stock?": "A blue-chip stock is a stock from a large, established, and financially sound company with a history of reliable performance.",
    "What is a small-cap stock?": "A small-cap stock is a stock from a company with a smaller market capitalization, typically more volatile but with higher growth potential.",
    "What is a credit bureau?": "A credit bureau is an agency that collects and maintains consumer credit information and provides it to lenders for assessing creditworthiness.",
    "What is a liability insurance?": "Liability insurance provides coverage for legal claims made against the insured, protecting against financial loss due to legal liability.",
    "What is a deductible?": "A deductible is the amount you pay out of pocket before your insurance coverage kicks in.",
    "What is a premium?": "A premium is the payment made to an insurance company in exchange for coverage.",
    "What is co-insurance?": "Co-insurance is the percentage of costs that the insured must pay after meeting the deductible.",
    "What is a beneficiary?": "A beneficiary is a person designated to receive assets or benefits from a trust, will, or insurance policy.",
    "What is an HSA (Health Savings Account)?": "An HSA is a tax-advantaged account used to save for medical expenses, available to those with high-deductible health plans.",
    "What is a 403(b) plan?": "A 403(b) plan is a retirement savings plan available to employees of certain public schools and tax-exempt organizations.",
    "What is a bond yield?": "Bond yield refers to the return an investor realizes on a bond, expressed as a percentage.",
    "What is a zero-coupon bond?": "A zero-coupon bond is a bond that does not pay periodic interest but is sold at a discount and matures at its face value.",
    "What is sovereign debt?": "Sovereign debt is the money borrowed by a government from foreign or domestic investors.",
    "What is a revolving credit?": "Revolving credit is a type of credit that allows the borrower to use or withdraw funds up to a certain limit, repay it, and borrow again.",
    "What is depreciation expense?": "Depreciation expense is the allocation of the cost of a tangible asset over its useful life.",
    "What is goodwill in accounting?": "Goodwill is an intangible asset that arises when a company acquires another business for more than the fair value of its net assets.",
    "What is a financial derivative?": "A financial derivative is a contract whose value is based on the performance of an underlying asset, such as stocks, bonds, or currencies.",
    "What is a futures contract?": "A futures contract is an agreement to buy or sell an asset at a future date at an agreed-upon price.",
    "What is an options contract?": "An options contract gives the holder the right, but not the obligation, to buy or sell an asset at a specified price within a certain period.",
    "What is a hedge?": "A hedge is an investment made to reduce the risk of adverse price movements in an asset.",
    "What is a financial covenant?": "A financial covenant is a condition in a loan agreement that the borrower must adhere to, such as maintaining certain financial ratios.",
    "What is operational risk?": "Operational risk refers to the risk of loss due to failures in internal processes, systems, or external events.",
    "What is a balance transfer?": "A balance transfer is the process of moving debt from one credit card to another, usually to take advantage of lower interest rates.",
    "What is a short sale?": "A short sale involves selling a security that the seller does not own, with the expectation of repurchasing it later at a lower price.",
    "What is crowdfunding?": "Crowdfunding is the practice of funding a project or venture by raising small amounts of money from a large number of people, typically via the internet.",
    "What is angel investing?": "Angel investing involves wealthy individuals investing their own funds into early-stage startups in exchange for equity.",
    "What is venture capital?": "Venture capital is funding provided to startups and small businesses with high growth potential in exchange for equity.",
    "What is a public offering?": "A public offering is the sale of securities to the general public, typically through an initial public offering (IPO).",
    "What is a financial bubble?": "A financial bubble occurs when the price of an asset rises significantly over its intrinsic value, followed by a sharp decline.",
    "What is stagflation?": "Stagflation is a situation where economic growth is stagnant but inflation and unemployment are high.",
    "What is a secured credit card?": "A secured credit card requires a deposit that acts as collateral and typically has lower credit limits.",
    "What is an unsecured credit card?": "An unsecured credit card does not require collateral and is issued based on creditworthiness.",
    "What is a trade deficit?": "A trade deficit occurs when a country's imports exceed its exports, leading to a negative trade balance.",
    "What is a trade surplus?": "A trade surplus occurs when a country's exports exceed its imports, resulting in a positive trade balance.",
    "What is monetary policy?": "Monetary policy refers to the actions taken by a central bank to control the money supply and interest rates to achieve macroeconomic goals.",
    "What is fiscal policy?": "Fiscal policy refers to government spending and tax policies used to influence economic conditions.",
    "What is quantitative easing?": "Quantitative easing is a monetary policy where a central bank buys securities to increase the money supply and encourage lending and investment.",
    "What is a financial contagion?": "Financial contagion is the spread of economic crises from one market or region to others.",
    "What is a sovereign wealth fund?": "A sovereign wealth fund is a state-owned investment fund composed of money generated by the government, often from surplus reserves.",
    "What is leverage ratio?": "A leverage ratio measures the level of debt used by a company relative to its equity or assets.",
    "What is currency depreciation?": "Currency depreciation is a decrease in the value of a currency relative to other currencies.",
    "What is currency appreciation?": "Currency appreciation is an increase in the value of a currency relative to other currencies.",
    "What is a credit line?": "A credit line is an arrangement between a financial institution and a customer that establishes a maximum loan balance that the lender will allow.",
    "What is a cash advance?": "A cash advance is a service that allows you to withdraw cash against your credit card’s line of credit.",
    "What is a brokerage account?": "A brokerage account is an investment account that allows you to buy and sell a variety of investments, such as stocks, bonds, and mutual funds.",
    "What is a custodial account?": "A custodial account is a savings account managed by an adult for a minor, who will gain control of the account at a certain age.",
    "What is a growth stock?": "A growth stock is a share in a company expected to grow at an above-average rate compared to other companies.",
    "What is a value stock?": "A value stock is a share in a company that is considered undervalued compared to its financial performance and has potential for growth.",
    "What is a market order?": "A market order is an order to buy or sell a security immediately at the best available price.",
    "What is a limit order?": "A limit order is an order to buy or sell a security at a specified price or better.",
    "What is a stop order?": "A stop order is an order to buy or sell a security once it reaches a certain price, known as the stop price.",
    "What is a trailing stop order?": "A trailing stop order is a type of stop order that moves with the price fluctuations of a security.",
    "What is a money transfer service?": "A money transfer service allows users to send money domestically or internationally through financial institutions or payment service providers."
}

# fine-tuning using similarity
from difflib import get_close_matches

def get_answer(question):
    question = question.lower()
    closest_match = get_close_matches(question, [q.lower() for q in financial_qna.keys()], n=1, cutoff=0.6)
    if closest_match:
        original_question = [key for key in financial_qna.keys() if key.lower() == closest_match[0]][0]
        return financial_qna[original_question]
    return "Sorry, the answer to this question is not available."

# stop when the user types 'stop'
while True:
    user_question = input("Please type your financial question (or type 'stop' to exit): ")
    if user_question.lower() == 'stop':
        print("Goodbye!")
        break
    answer = get_answer(user_question)
    print(answer)


# In[ ]:




