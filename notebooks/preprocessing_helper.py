import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer



def clean_data(df, columns_to_include,not_null_columns):
        df = df[df.columns.intersection(columns_to_include)]
        df = df.dropna(subset=df.columns.intersection(not_null_columns))
        date_columns = ['bankstatementcreationdate','xeroreportcreationdate'] 
        object_columns = list(df.select_dtypes(include=['object']).columns)
        categorical_columns = ['quote', 'directorpropertytyped1','directorpropertytyped2', 'directorpropertytyped3', 'orgtype', 'industryclassification']
        for i in list(set(object_columns) - set(date_columns)):
                df[i] = df[i].apply(lambda x: str(x).replace(',','').replace('%','').replace('_',' ').replace('nan','').replace(' ','').lower())
                if i not in list(set(categorical_columns)):
                        df[i]  = pd.to_numeric(df[i], errors='coerce')
        return df



def date_string_to_day(x, level):
        x = pd.to_datetime(x)
        if level == 'month':
            return x.dt.day
        elif level == 'year':
            return x.dt.dayofyear
        return x

def data_normalization_based_on_date(df, date_column, columns_to_process, level = 'month'):
    df['day'] = df[date_column].apply(lambda x:  date_string_to_day(x, level), axis=1)
    df[columns_to_process] = df[columns_to_process].apply(lambda x:  (x/df["day"]), axis=0)
    df = df.drop(columns = ['day'], errors='ignore')
    return df



def aggregate(df, columns_to_aggregate, aggregated_column_prefix, method, drop_original = True):
    columns_to_aggregate = df.columns.intersection(columns_to_aggregate)
    aggregated_column = aggregated_column_prefix + method
    if (len(columns_to_aggregate) == 0):
        return df
    elif (len(columns_to_aggregate) == 1):
        df[aggregated_column] = df[columns_to_aggregate]
    elif method == 'sum':
        df[aggregated_column] = df[columns_to_aggregate].sum(axis=1, skipna = True)
    elif method == 'mean':
        df[aggregated_column] = df[columns_to_aggregate].mean(axis=1, skipna = True)
    elif method == 'max':
        df[aggregated_column] = df[columns_to_aggregate].max(axis=1, skipna = True)
    elif method == 'min':  
        df[aggregated_column] = df[columns_to_aggregate].min(axis=1, skipna = True)  
    if drop_original:
        df = df.drop(columns = columns_to_aggregate, errors='ignore')
    return df

def generating_features_prior_filling_gaps(df):
    
    columns_to_aggregate_temp = ["consumercreditjudgementsguar1",
                                                             "consumercreditjudgementsguar2",
                                                             "consumercreditjudgementsguar3"]
    df = aggregate(df, columns_to_aggregate_temp, "consumercreditjudgementsguar","sum")

    columns_to_aggregate_temp = ["consumercreditinsolvencynoticesguar1",
                                                             "consumercreditinsolvencynoticesguar2",
                                                             "consumercreditinsolvencynoticesguar3"]
    df = aggregate(df, columns_to_aggregate_temp, "consumercreditinsolvencynoticesguar","sum")

    columns_to_aggregate_temp = ["consumercreditcreditdefaultsguar1",
                                                             "consumercreditcreditdefaultsguar2",
                                                             "consumercreditcreditdefaultsguar3"]
    df = aggregate(df, columns_to_aggregate_temp, "consumercreditcreditdefaultsguar","sum")

    columns_to_aggregate_temp = ["consumercreditcompanyaffiliationsguar1",
                                                             "consumercreditcompanyaffiliationsguar2",
                                                             "consumercreditcompanyaffiliationsguar3"]
    df = aggregate(df, columns_to_aggregate_temp, "consumercreditcompanyaffiliationsguar","mean")

    columns_to_aggregate_temp = ["consumercreditfileactivityguar1",
                                                             "consumercreditfileactivityguar2",
                                                             "consumercreditfileactivityguar3"]
    df = aggregate(df, columns_to_aggregate_temp, "consumercreditfileactivityguar","min",False)
    df = aggregate(df, columns_to_aggregate_temp, "consumercreditfileactivityguar","mean",False)
    df = aggregate(df, columns_to_aggregate_temp, "consumercreditfileactivityguar","max")


    columns_to_aggregate_temp = ["consumercreditriskoddsguar1",
                                                             "consumercreditriskoddsguar2",
                                                             "consumercreditriskoddsguar3"]
    df = aggregate(df, columns_to_aggregate_temp, "consumercreditriskoddsguar","min",False)
    df = aggregate(df, columns_to_aggregate_temp, "consumercreditriskoddsguar","mean",False)
    df = aggregate(df, columns_to_aggregate_temp, "consumercreditriskoddsguar","max")

    
    
    columns_to_aggregate_temp = [ "directorpropertyrateablevalued1",
                                                        "directorpropertyrateablevalued2",
                                                        "directorpropertyrateablevalued3"
                                                        ]
    df = aggregate(df, columns_to_aggregate_temp, "directorpropertyrateablevalued","sum")



    columns_to_aggregate_temp = ["accountspayabletotalsixmonthspast",
                                                                "accountspayabletotalsevenmonthspast",
                                                                "accountspayabletotaleightmonthspast",
                                                                "accountspayabletotalninemonthspast",
                                                                "accountspayabletotaltenmonthspast",
                                                                "accountspayabletotalelevenmonthspast",
                                                                "accountspayabletotaltwelvemonthspast"]

    df = aggregate(df, columns_to_aggregate_temp, "accountspayabletotal6to12months","sum")

    columns_to_aggregate_temp =["accountsreceivabletotalsixmonthspast",
                                                                "accountsreceivabletotalsevenmonthspast",
                                                                "accountsreceivabletotaleightmonthspast",
                                                                "accountsreceivabletotalninemonthspast",
                                                                "accountsreceivabletotaltenmonthspast",
                                                                "accountsreceivabletotalelevenmonthspast",
                                                                "accountsreceivabletotaltwelvemonthspast"]

    df = aggregate(df, columns_to_aggregate_temp, "accountsreceivabletotal6to12months","sum")

    return df




   
def fill_missing_values(df):

    columns_fill_with_previous = ["pltotalincome",
                                    "plgrossprofit",
                                    "pltotalotherincome",
                                    "pltotaloperatingexpenses",
                                    "plnetprofit"]

    columns_fill_with_previous_levels = ["current", "yr1","yr2","yr3"]

    for level in range(1,len(columns_fill_with_previous_levels)):
        columns_previous =list(map(lambda x: x + columns_fill_with_previous_levels[level-1] , columns_fill_with_previous))
        columns_to_fill =list(map(lambda x: x + columns_fill_with_previous_levels[level] , columns_fill_with_previous))
        df[columns_to_fill] = df[columns_previous]

    
    cat_features = list(df.select_dtypes(include=['object']).columns)
    float_features = list(df.select_dtypes(include=['float']).columns)
    
    num_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 0)
    cat_imputer1 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = "unknown")
    cat_imputer2 = SimpleImputer(missing_values='none', strategy='constant', fill_value = "unknown")
    cat_imputer3 = SimpleImputer(missing_values='null', strategy='constant', fill_value = "unknown")
    
    
    df[float_features]=num_imputer.fit_transform(df[float_features])
    df[cat_features] = cat_imputer1.fit_transform(df[cat_features])
    df[cat_features] = cat_imputer2.fit_transform(df[cat_features])
    df[cat_features] = cat_imputer3.fit_transform(df[cat_features])
    
    return df


def generating_descriptive_features(df):
    df["total_recieve_vs_borrow"] = (1+30*df["banksummarytotalreceivedcurrent"])/(1+df["fundtaploanprincipal"])
    df["total_recieve_vs_spend"] = (1+df["banksummarytotalreceivedcurrent"])/(1+df["banksummarytotalspentcurrent"])
    df["funded_outstanding"] = df["priorfundtaphistoryfundedsum"] - df["priorfundtaphistorycompletedsum"]
    df["liability_net_profit_ratio"] = ((df["execsummarynetassetscurrent"]+1)/(df["execsummarycurrentassetstoliabilitiescurrent"]+1))/(df["plnetprofitcurrent"]+1)
    df["net_profit_total_income_ratio"] = df["plnetprofitcurrent"]/ df["pltotalincomecurrent"]
    df["net_profit_net_asset_ratio"] = (1+df["plnetprofitcurrent"])/ (1+df["execsummarynetassetscurrent"])
    df["concentration_risk_current"] = df["funded_outstanding"]/(df["accountsreceivabletotalcurrentmonth"]+1)
    df["concentration_risk_m1"] = df["funded_outstanding"]/(df["accountsreceivabletotalonemonthpast"]+1)
    df["concentration_risk_m2"] = df["funded_outstanding"]/(df["accountsreceivabletotaltwomonthspast"]+1)
    df["concentration_risk_m3"] = df["funded_outstanding"]/(df["accountsreceivabletotalthreemonthspast"]+1)
    df["concentration_risk_m4"] = df["funded_outstanding"]/(df["accountsreceivabletotalfourmonthspast"]+1)
    return df


def preprocessing(df, method = "binary"):
    columns_to_include = [ 'consumercreditinsolvencynoticesguar1',
    'consumercreditcreditdefaultsguar1',
    'consumercreditcompanyaffiliationsguar1',
    'consumercreditfileactivityguar1',
    'consumercreditriskoddsguar1',
    'consumercreditcreditdefaultsguar2',
    'consumercreditcompanyaffiliationsguar2',
    'consumercreditfileactivityguar2',
    'consumercreditriskoddsguar2',
    'consumercreditcreditdefaultsguar3',
    'consumercreditcompanyaffiliationsguar3',
    'consumercreditfileactivityguar3',
    'consumercreditriskoddsguar3',
    'commercialcreditdirectors',
    'commercialcreditjudgements',
    'commercialcreditcreditactivity',
    'commercialcreditppsr',
    'commercialcreditpublicnotices',
    'commercialcreditscore',
    'commercialcreditcreditlimit',
    'commercialcreditsuppliers',
    'commercialcreditdsocurrent',
    'commercialcreditdsoonemonth',
    'commercialcreditdsotwomonths',
    'directorpropertytyped1',
    'directorpropertyrateablevalued1',
    'directorpropertytyped2',
    'directorpropertyrateablevalued2',
    'directorpropertytyped3',
    'directorpropertyrateablevalued3',
    'orgtype',
    'industryclassification',
    'priorfundtaphistorycompletedsum',
    'priorfundtaphistoryfundedsum',
    'priorfundtaphistoryduesum',
    'pltotalincomecurrent',
    'plgrossprofitcurrent',
    'pltotalotherincomecurrent',
    'pltotaloperatingexpensescurrent',
    'plnetprofitcurrent',
    'pltotalincomeyr1',
    'plgrossprofityr1',
    'pltotalotherincomeyr1',
    'pltotaloperatingexpensesyr1',
    'plnetprofityr1',
    'pltotalincomeyr2',
    'plgrossprofityr2',
    'pltotalotherincomeyr2',
    'pltotaloperatingexpensesyr2',
    'plnetprofityr2',
    'pltotalincomeyr3',
    'plgrossprofityr3',
    'pltotalotherincomeyr3',
    'pltotaloperatingexpensesyr3',
    'plnetprofityr3',
    'accountspayabletotalcurrentmonth',
    'accountspayabletotalonemonthpast',
    'accountspayabletotaltwomonthspast',
    'accountspayabletotalthreemonthspast',
    'accountspayabletotalfourmonthspast',
    'accountspayabletotalfivemonthspast',
    'accountspayabletotalsixmonthspast',
    'accountspayabletotalsevenmonthspast',
    'accountspayabletotaleightmonthspast',
    'accountspayabletotalninemonthspast',
    'accountspayabletotaltenmonthspast',
    'accountspayabletotalelevenmonthspast',
    'accountsreceivabletotalcurrentmonth',
    'accountsreceivabletotalonemonthpast',
    'accountsreceivabletotaltwomonthspast',
    'accountsreceivabletotalthreemonthspast',
    'accountsreceivabletotalfourmonthspast',
    'accountsreceivabletotalfivemonthspast',
    'accountsreceivabletotalsixmonthspast',
    'accountsreceivabletotalsevenmonthspast',
    'accountsreceivabletotaleightmonthspast',
    'accountsreceivabletotalninemonthspast',
    'accountsreceivabletotaltenmonthspast',
    'accountsreceivabletotalelevenmonthspast',
    'banksummarytotalspentcurrent',
    'banksummarytotalreceivedcurrent',
    'banksummarytotalspentm1',
    'banksummarytotalreceivedm1',
    'banksummarytotalspentm2',
    'banksummarytotalreceivedm2',
    'banksummarytotalspentm3',
    'banksummarytotalreceivedm3',
    'execsummarycashreceivedcurrent',
    'execsummarycashspentcurrent',
    'execsummarycashsurplusdeficitcurrent',
    'execsummaryclosingbankbalancecurrent',
    'execsummaryincomecurrent',
    'execsummarydirectcostscurrent',
    'execsummarygrossprofitlosscurrent',
    'execsummaryotherincomecurrent',
    'execsummaryexpensescurrent',
    'execsummaryprofitlosscurrent',
    'execsummarydebtorscurrent',
    'execsummarycreditorscurrent',
    'execsummarynetassetscurrent',
    'execsummarynumberofinvoicesissuedcurrent',
    'execsummaryaveragevalueofinvoicescurrent',
    'execsummarygrossprofitmargincurrent',
    'execsummarynetprofitmargincurrent',
    'execsummaryreturnoninvestmentcurrent',
    'execsummaryaveragedebtorsdayscurrent',
    'execsummaryaveragecreditorsdayscurrent',
    'execsummaryshorttermcashforecastcurrent',
    'execsummarycurrentassetstoliabilitiescurrent',
    'execsummarytermassetstoliabilitiescurrent',
    'execsummarycashreceivedm1',
    'execsummarycashspentm1',
    'execsummarycashsurplusdeficitm1',
    'execsummaryclosingbankbalancem1',
    'execsummaryincomem1',
    'execsummarydirectcostsm1',
    'execsummarygrossprofitlossm1',
    'execsummaryotherincomem1',
    'execsummaryexpensesm1',
    'execsummaryprofitlossm1',
    'execsummarydebtorsm1',
    'execsummarycreditorsm1',
    'execsummarynetassetsm1',
    'execsummarynumberofinvoicesissuedm1',
    'execsummaryaveragevalueofinvoicesm1',
    'execsummarygrossprofitmarginm1',
    'execsummarynetprofitmarginm1',
    'execsummaryreturnoninvestmentm1',
    'execsummaryaveragedebtorsdaysm1',
    'execsummaryaveragecreditorsdaysm1',
    'execsummaryshorttermcashforecastm1',
    'execsummarycurrentassetstoliabilitiesm1',
    'execsummarytermassetstoliabilitiesm1',
    'fundtaploanprincipal',
    'customeruid',
    'bankstatementcreationdate',
    'xeroreportcreationdate',
    'priorfundtaphistorypendingsum',
    'quote',
    'fundtapprofitloss']
    if method == "multi":
        columns_to_include.append("weekspastdue")
    not_null_columns = ['bankstatementcreationdate','xeroreportcreationdate'] 
    df = clean_data(df, columns_to_include, not_null_columns)
        
    if 'bankstatementcreationdate' in df.columns:
        bank_date_colum = ['bankstatementcreationdate']
        bank_columns_to_process= ["banksummarytotalspentcurrent","banksummarytotalreceivedcurrent"]
        df = data_normalization_based_on_date(df, bank_date_colum,bank_columns_to_process)
        df = df.drop(columns = bank_date_colum, errors='ignore')

    if 'xeroreportcreationdate' in df.columns:
        xero_date_colum = ['xeroreportcreationdate']
        xero_columns_to_process= ["accountspayabletotalcurrentmonth",
                                            "accountsreceivabletotalcurrentmonth", 
                                            "execsummarycashreceivedcurrent",
                                            "execsummarycashspentcurrent",
                                            "execsummarycashsurplusdeficitcurrent",
                                            "execsummaryclosingbankbalancecurrent",
                                            "execsummaryincomecurrent",
                                            "execsummarydirectcostscurrent",
                                            "execsummarygrossprofitlosscurrent",
                                            "execsummaryotherincomecurrent",
                                            "execsummaryexpensescurrent",
                                            "execsummaryprofitlosscurrent",
                                            "execsummarydebtorscurrent",
                                            "execsummarycreditorscurrent",
                                            "execsummarynetassetscurrent",
                                            "execsummarynumberofinvoicesissuedcurrent",
                                            "execsummaryaveragevalueofinvoicescurrent",
                                            "execsummarygrossprofitmargincurrent",
                                            "execsummarynetprofitmargincurrent",
                                            "execsummaryreturnoninvestmentcurrent",
                                            "execsummaryaveragedebtorsdayscurrent",
                                            "execsummaryaveragecreditorsdayscurrent",
                                            "execsummaryshorttermcashforecastcurrent",
                                            "execsummarycurrentassetstoliabilitiescurrent",
                                            "execsummarytermassetstoliabilitiescurrent"]
        df = data_normalization_based_on_date(df, xero_date_colum,xero_columns_to_process)

        
        xero_columns_to_process = ["pltotalincomecurrent",
                                            "plgrossprofitcurrent",
                                            "pltotalotherincomecurrent",
                                            "pltotaloperatingexpensescurrent",
                                            "plnetprofitcurrent"]
        df = data_normalization_based_on_date(df, xero_date_colum,xero_columns_to_process, level = 'year')
        df = df.drop(columns = xero_date_colum, errors='ignore')

    
    df = generating_features_prior_filling_gaps(df)
    df = fill_missing_values(df)
    df = generating_descriptive_features(df)

    return df
