import warnings
import scipy.stats as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.tsa.stattools import adfuller, acf, pacf, arma_order_select_ic, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import itertools
from statsmodels.tsa.ar_model import AR
from arch import arch_model
#from tqdm import tqdm


class CATS():

    def __init__(self, series, data_window=None, pbar=False):
        self.pbar = pbar
        if data_window:
            self.series = series[-data_window:]
        else:
            self.series = series
        diff = series - series.shift()
        self.diff = diff.dropna()
        self.diff2 = series.diff().dropna().diff().dropna()
        self.forecast_tests = {}
        self.grids = {}
        self.resid = {}
        self.residiid = {}
        self.engle = {}
        self.standardiid = {}
        self.standard_engle = {}
        self.nonseasonal_result = np.NaN
        self.seasonal_result = np.NaN
        self.bestmodel = [np.NaN, np.NaN, np.NaN, np.NaN, np.Inf]

        """
        Attribute
        -------------
        pbar: Boolean to activate progress bar when running models.
        series: Input time series.
        diff: Differenced of input time series.
        diff2: 2 difference of input time series.

        Keys in following dictionary: "ARIMA(1)", "ARIMA(2)", "HoltWinters", "SARIMA(1)", "SARIMA(2)", "Seasonal_HoltWinters"
        forecast_tests: Dictionary of dataframes of forecast test ran.
        grids: Dictionary of grid search dataframe for all model parameter grid searches (except GARCH).
        resid: Dictionary containing residual series of all the models
        residiid: Dictionary containing test statistics and pvalue of SampleACF test and Portmanteau test for model residuals.
        value = {"SampleACF":[boolean, number of exceeds, threshold of exceed], "Portmanteau":[boxpiearce_boolean, boxpierce stats, ljungbox_boolean, ljungbox stats]} 95%critical value: 55.75847927888702

        engle: Dictionary containing engle test pvalue of model residual.
        standardiid:  Dictionary containing test statistics and pvalue of SampleACF test and Portmanteau test for GARCH standardised residuals. Same values as residiid.
        standard_engle: Dictionary containing engle test pvalue of GARCH standardised residual.
        nonseasonal_result: Dataframe of non_seasonal search.
        seasonal_result: Dataframe of seasonal search.
        bestmodel: [name of model, modelparameter, model aic, garch parameter, mape] of fitted best model with lowest mape.

        """

    def test_stationarity(self, ts, disp=True, window=60):
        onerolmean = ts.rolling(window=window).mean()
        onerolstd = ts.rolling(window=window).std()

        if disp:
            plt.figure(figsize=(14,6))
            plt.plot(ts, color= "black", label= "actual")
            plt.plot(onerolmean, color="red", label="rolling mean")
            plt.plot(onerolstd, color="blue", label="rolling STD")
            plt.legend()
            plt.show()
        
        dftest = adfuller(ts, autolag="AIC")

        dftestprint = pd.Series(dftest[:4], index=["Test Statistic:", "p-value:", "#Lags Used:", "#Observations:"])
        for key, value in dftest[4].items():
            dftestprint["Critical Value (%s):"%key] = value
        
        if disp:
            print("Augmented Dickey-Fuller Test")
            print("H0: Unit Root is present in time series and thus it is non-stationary")
            print("H1: Time series is stationary \n")
            print(dftestprint, "\n")
            print("Reject null hypothesis when test statistic < critical values and p-value < levels", "\n")

        if (dftestprint[0] < dftestprint[5]):
            if disp:
                print("At 5% level, H0 rejected, series is Stationary")
            else:
                return [True, [ts, onerolmean, onerolstd]]
        else:
            if disp:
                print("At 5% level, H0 not rejected, Unit Root present (series is non-stationary)")
            else:
                return [False, [ts,onerolmean, onerolstd]]

    
    def test_trendstationarity(self, disp=True):
        dftest = adfuller(self.series, autolag="AIC", regression="ct")
        dftestprint = pd.Series(dftest[:4], index=["Test Statistic:", "p-value:", "#Lags Used:", "#Observations:"])
        for key, value in dftest[4].items():
            dftestprint["Critical Value (%s):"%key] = value
        kpsstest = kpss(self.series, regression="ct")
        kpssprint = pd.Series(kpsstest[:3], index=["Test Statistic:", "p-value (actual p-value smaller):", "#Lags Used:"])
        for key, value in kpsstest[3].items():
            kpssprint["Critical Value (%s):"%key] = value

        if disp:    
            print("Augmented Dickey-Fuller Test")
            print("H0: Unit Root is present in time series and thus it has non-stationary trend")
            print("H1: Time series is trend stationary \n")
            print(dftestprint)
            print("Reject null hypothesis when test statistic < critical values and p-value < levels \n")
            
            print("KPSS Test")
            print("H0: Time series is trend stationary")
            print("H1: Unit Root is present in time series and thus it has non-stationary trend \n")
            print(kpssprint)
            print("Reject null hypothesis when test statistic > critical values and p-value < levels \n")
        
            print("Summary at 5% significance level")

        noti = list()
        if dftest[0] < dftest[4]["5%"]:
            if disp:
                print("Augmented Dickey-Fuller Test, series is trend stationary (H0 rejected)")
            noti.append("Augmented Dickey-Fuller Test: H0 rejected, series is Trend Stationary")
        else:
            if disp:
                print("Augmented Dickey-Fuller Test, series has non-stationary trend (H0 not rejected)")
            noti.append("Augmented Dickey-Fuller Test: H0 not rejected, series has non-stationary trend")
            
        if kpsstest[0] > kpsstest[3]["5%"]:
            if disp:
                print("KPSS Test, series has non-stationary trend (H0 rejected)")
            noti.append("KPSS Test: H0 rejected, series has non-stationary trend")
        else:
            if disp:
                print("KPSS Test, series is trend stationary (H0 not rejected)")
            noti.append("KPSS Test: H0 not rejected, series is Trend Stationary")

        if disp == False:
            return noti

    def visualise_season(self, period):
        toadd = period - (len(self.series.get_values()) % period)
        arr = np.append(self.series.sort_index().get_values(), (np.zeros(toadd) + np.NaN)).reshape((-1, period))
        seasondf = pd.DataFrame(arr).transpose()
        plt.figure(figsize=(20,6))
        plt.plot(seasondf)
        plt.xlabel("Seasonal Period of "+str(period))
        plt.show()

    def analysis(self, window=60, season=[60,]):
        station_res = self.test_stationarity(self.series, disp=False, window=window)
        trendstation_res = self.test_trendstationarity(disp=False)

        plt.figure(figsize=(14,6))
        plt.plot(station_res[1][0], color= "black", label= "actual")
        plt.plot(station_res[1][1], color="red", label="rolling mean")
        plt.plot(station_res[1][2], color="blue", label="rolling STD")
        plt.legend()
        plt.show()

        print("Stationarity Test (at 5% level)")
        if station_res[0]:
            print("Augmented Dickey-Fuller Test: H0 rejected, series is Stationary")
        else:
            print("Augmented Dickey-Fuller Test: H0 not rejected, Unit Root present (series is non-stationary)")
        print("------------------------------------------------------------------------------------------------", "\n")
        print("Trend Stationarity Tests (at 5% level)")
        print(trendstation_res[0])
        print(trendstation_res[1])
        print("------------------------------------------------------------------------------------------------", "\n")
        print("Seasonality Visualisation:")
        for s in season:
            self.visualise_season(s)

    def grid_arma(self, ts):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            gridarma = arma_order_select_ic(ts, ic="aic", max_ma=4)
            
            armadict = {}

            if self.pbar:
                from tqdm import tqdm
                itergrid = tqdm(list(gridarma["aic"].columns), desc="ARIMA Grid Search")
            else:
                itergrid = list(gridarma["aic"].columns)
            
            for q in itergrid:
                for p in range(len(gridarma["aic"][q])):
                    armadict[(p,q)] = [gridarma["aic"][q][p]]
            gridarmadf = pd.DataFrame(armadict)
            gridarmadf = gridarmadf.transpose()
            param = list(gridarmadf.index)
            gridarmadf["param"] = param
            gridarmadf = gridarmadf.rename(columns={0: "aic"})
            gridarmadf = gridarmadf.reset_index(drop=True)
            gridarmadf = gridarmadf.sort_values(by="aic")
            return gridarmadf

    #have not check for error to catch it model do not fit during forecast
    def onestep_forecast(self, ts, totalnum_test, model=ARIMA, **kwargs):
        df_dict = {"actual":[], "predict":[], "difference":[], "date":[]}
        ts = ts.sort_index()

        if self.pbar:
            from tqdm import tqdm
            iterfc = tqdm(range(totalnum_test), desc="Forecast Test")
        else:
            iterfc = range(totalnum_test)
        for i in iterfc:
            index = totalnum_test - i
            train = ts[:-index]
            actual = ts[-index]
            date = ts.index[-index]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                inimodel = model(train, **kwargs)    
                fcfit = inimodel.fit()
                if model == ARIMA:
                    predict = fcfit.forecast(1)[0]
                else:
                    predict = fcfit.forecast(1)
            #predict = fcfit.forecast(1)[0]
            predict = np.array(predict)
            diff = actual - predict[0]
            df_dict["date"].append(date)
            df_dict["difference"].append(diff)
            df_dict["predict"].append(predict[0])
            df_dict['actual'].append(actual)
            
            #print(str(i+1)+"/"+str(totalnum_test), "done")
        newdf = pd.DataFrame(df_dict)
        newdf = newdf.set_index(["date"])
        return newdf

    def check_residualacf(self, residual, lags=40):
        acfdata = acf(residual, nlags=lags)

        up = sum(acfdata > 1.96/sqrt(len(residual)))
        down = sum(acfdata < -1.96/sqrt(len(residual)))
        total = up + down
        num_CI = lags - lags/100*95

        #check if residual iid if total > num_CI, residuals NOT iid
        if total > num_CI:
            return [False, total, num_CI]
        else:
            return [True, total, num_CI]

    def portmanteau(self, residual, acf_nlags=40):
        statslst = acorr_ljungbox(residual, lags=acf_nlags, boxpierce=True)

        bp_stats = statslst[2][acf_nlags-1]
        bp_pvalue = statslst[3][acf_nlags-1]
        lb_stats = statslst[0][acf_nlags-1]
        lb_pvalue = statslst[1][acf_nlags-1]
        
        chi2stats95 = st.chi2.ppf(0.95, acf_nlags)

        lst = []

        if bp_stats > chi2stats95:
            lst.extend([False, bp_pvalue])
        else:
            lst.extend([True, bp_pvalue])

        if lb_stats > chi2stats95:
            lst.extend([False, lb_pvalue])
        else:
            lst.extend([True, lb_pvalue])

        lst.extend([chi2stats95])
        return lst

    def engle_archtest(self, residual):
        info = het_arch(residual)

        #if p-value < 0.05, at 5% level, reject H0, arch effect is present
        if info[1] < 0.05:
            return [True, info[1]]
        else:
            return [False, info[1]]

    def grid_garch(self, residual, r_range=(1,4), s_range=(0,4), garch_dist="t"):
    
        # Define the r and s parameters to take any value between 0 and 4
        r = range(r_range[0], r_range[1])
        s = range(s_range[0], s_range[1])

        # Generate all different combinations of p, q and q triplets
        rs = list(itertools.product(r, s))
        
        dfdict = {}
        for order in rs:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = arch_model(residual ,mean="constant", p=order[0], q=order[1], dist=garch_dist)
                res = model.fit(disp="off")
            aic = res.aic
            bic = res.bic
            dfdict[order] = [aic, bic]
        
        newdf = pd.DataFrame(dfdict, index=["aic", "bic"])
        sortdf = newdf.transpose().sort_values(by="aic")
        param = list(sortdf.index)
        sortdf["param"] = param
        sortdf = sortdf.reset_index(drop=True)
        
        bestaic = sortdf["aic"][0]
        bestpara = sortdf.index[0]
        
        return sortdf

    def garch_search(self, residual, garch_dist="t"):
        bestgarchfit = None
        bestgaram = None
                    
        gridgarch = self.grid_garch(residual)

        for garam in gridgarch["param"]:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    garchmodel = arch_model(residual, mean="constant",p=garam[0], q=garam[1], vol="GARCH", dist=garch_dist)
                    garchfit = garchmodel.fit(disp='off')

                    bestgarchfit = garchfit
                    bestgaram = garam
                    break
            except:
                continue

        if bestgaram:
            #check if standardised residuals after garch are iid
            #standres = residual / bestgarchfit.conditional_volatility
            standres = bestgarchfit.resid / bestgarchfit.conditional_volatility

            stanacfcheck = self.check_residualacf(standres)
            standport = self.portmanteau(standres)
            standengle = self.engle_archtest(standres)

            standnumpass = sum([stanacfcheck[0], standport[0], standport[2]])

            #set if 2 or more test shows NOT iid, indicate that standardised residuals are not iid
            return [bestgaram, [standnumpass>=2, {"SampleACF": stanacfcheck, "Portmanteau": standport}], standengle]

        #event where no parameters fit garch
        else:
            return [np.NaN, [np.NaN, np.NaN], np.NaN]

    def arimagarch_search(self, difference=1, fc_numtest=20, resid_lags=40, garch_dist="t"):
        if difference==0:
            ts=self.series
        elif difference == 1:
            ts = self.diff
        elif difference == 2:
            ts = self.diff2
        else:
            return "Only ARIMA of maximum difference 2 is accepted"

        gridarmadf = self.grid_arma(ts)
        dlab = "ARIMA("+str(difference)+")"
        self.grids[dlab] = gridarmadf

        bestparam = None
        bestarifit = None

        for param in gridarmadf["param"]:
            param = list(param)
            param.insert(1,difference)
            param = tuple(param)

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    model = ARIMA(self.series, order=param)
                    arimafit = model.fit()

                    bestarifit = arimafit
                    bestparam = param
                    break
            except:
                continue

        if bestparam:
            arifc = self.onestep_forecast(self.series, fc_numtest, order=bestparam)
            self.forecast_tests[dlab] = arifc
            
            mape = np.mean(abs(arifc["actual"] - arifc["predict"])) / np.mean(arifc["actual"]) * 100
            fluc = np.mean(arifc["actual"].diff())
            maefluc = np.mean(abs(arifc["actual"] - arifc["predict"])) / fluc * 100

            self.resid[dlab] = bestarifit.resid
            #check if residuals are iid
            residacf = self.check_residualacf(bestarifit.resid, lags=resid_lags)
            residport = self.portmanteau(bestarifit.resid, acf_nlags=resid_lags)
            self.residiid[dlab] = {"SampleACF": residacf, "Portmanteau": residport}

            numpass = sum([residacf[0], residport[0], residport[2]])

            #set if 2 or more test shows NOT iid, indicate that residuals are not iid
            if numpass <= 1:
                #check for arch effect
                archind = self.engle_archtest(bestarifit.resid)
                self.engle[dlab] = archind

                # if arch effect present, find best fit garch model, garch_search returns [best garch param, if standardised resid iid, if arch effect present in standardised resid]    
                if archind[0]:

                    garch_res = self.garch_search(bestarifit.resid, garch_dist=garch_dist)
                    self.standardiid[dlab] = garch_res[1][1]
                    self.standard_engle[dlab] = garch_res[-1]

                    res = [bestparam, bestarifit.aic, numpass>=2, mape, maefluc, archind[0], garch_res[0], garch_res[1][0], garch_res[2][0]] 
                    return res

                #event where residual not iid and have no arch effect
                else:
                    return [bestparam, bestarifit.aic, numpass>=2, mape, maefluc, archind[0], np.NaN, np.NaN, np.NaN]

            #event where arima has iid residuals
            else:
                return [bestparam, bestarifit.aic, numpass>=2, mape, maefluc, np.NaN, np.NaN, np.NaN, np.NaN]

        #event where no parameters fit arima model
        else:
            return [np.NaN] * 9

    def holt_grid(self, checkdamped=True, seasonal_periods=None):
        options = ["add", "mul"]
        dampedchoice = [True, False]
        dfdict = {}

        if self.pbar:
            from tqdm import tqdm
            itergrid = tqdm(options, desc="HoltWinters Grid Search")
        else:
            itergrid = options
        
        for op in itergrid:
            if checkdamped:
                for c in dampedchoice:
                    if seasonal_periods:
                        for s in options:
                            for p in seasonal_periods:
                                if c:
                                    lab = ((op, "damped"), (s, p))
                                else:
                                    lab = ((op,), (s, p))
                                hfit = ExponentialSmoothing(self.series ,trend=op, damped=c, seasonal=s, seasonal_periods=p).fit()
                                rmse = sqrt(np.mean((self.series - hfit.fittedvalues) **2))
                                bic = hfit.bic
                                aic = hfit.aic
                                dfdict[lab] = [aic, bic, rmse]
                                
                    else:
                        if c:
                            lab = ((op, "damped"),)
                        else:
                            lab = ((op,),)
                        hfit = ExponentialSmoothing(self.series ,trend=op, damped=c).fit()
                        rmse = sqrt(np.mean((self.series - hfit.fittedvalues) **2))
                        bic = hfit.bic
                        aic = hfit.aic
                        dfdict[lab] = [aic, bic, rmse]
                        
            else:
                if seasonal_periods:
                    for s in options:
                        for p in seasonal_periods:                                
                            lab = ((op,), (s, p))
                            hfit = ExponentialSmoothing(self.series ,trend=op, seasonal=s, seasonal_periods=p).fit()
                            rmse = sqrt(np.mean((self.series - hfit.fittedvalues) **2))
                            bic = hfit.bic
                            aic = hfit.aic
                            dfdict[lab] = [aic, bic, rmse]
                            
                else:
                    lab = ((op,),)
                    hfit = ExponentialSmoothing(self.series ,trend=op).fit()
                    rmse = sqrt(np.mean((self.series - hfit.fittedvalues) **2))
                    bic = hfit.bic
                    aic = hfit.aic
                    dfdict[lab] = [aic, bic, rmse]
                    
        newdf = pd.DataFrame(dfdict, index=["aic", "bic", "RMSE Values"])
        sortdf =newdf.transpose().sort_values(by="aic")
        param = list(sortdf.index)
        sortdf["param"] = param
        sortdf = sortdf.reset_index(drop=True)
        return sortdf

    def holtgarch(self,checkdamped=True, seasonal_periods=None, fc_numtest=20, resid_lags=40, garch_dist="t"):

        gridholt = self.holt_grid(checkdamped=checkdamped, seasonal_periods=seasonal_periods)
        if seasonal_periods:
            self.grids["Seasonal_HoltWinters"] = gridholt
        else:
            self.grids["HoltWinters"] = gridholt

        if seasonal_periods:
            bestparams = gridholt["param"][0]

            if len(bestparams[0]) == 2:
                holtfit = ExponentialSmoothing(self.series ,trend=bestparams[0][0], damped=True, seasonal=bestparams[1][0], seasonal_periods=bestparams[1][1]).fit()
                holtfc = self.onestep_forecast(self.series, fc_numtest, model=ExponentialSmoothing, trend=bestparams[0][0], damped=True, seasonal=bestparams[1][0], seasonal_periods=bestparams[1][1])
                self.forecast_tests["Seasonal_HoltWinters"] = holtfc
                mape = np.mean(abs(holtfc["actual"] - holtfc["predict"])) / np.mean(holtfc["actual"]) * 100
                fluc = np.mean(holtfc["actual"].diff())
                maefluc = np.mean(abs(holtfc["actual"] - holtfc["predict"])) / fluc * 100
            else:
                holtfit = ExponentialSmoothing(self.series ,trend=bestparams[0][0], seasonal=bestparams[1][0], seasonal_periods=bestparams[1][1]).fit()
                holtfc = self.onestep_forecast(self.series, fc_numtest, model=ExponentialSmoothing, trend=bestparams[0][0], seasonal=bestparams[1][0], seasonal_periods=bestparams[1][1])
                self.forecast_tests["Seasonal_HoltWinters"] = holtfc
                mape = np.mean(abs(holtfc["actual"] - holtfc["predict"])) / np.mean(holtfc["actual"]) * 100
                fluc = np.mean(holtfc["actual"].diff())
                maefluc = np.mean(abs(holtfc["actual"] - holtfc["predict"])) / fluc * 100

        else:
            bestparams = gridholt["param"][0][0]
            if len(bestparams) == 2:
                holtfit = ExponentialSmoothing(self.series ,trend=bestparams[0], damped=True).fit()
                holtfc = self.onestep_forecast(self.series, fc_numtest, model=ExponentialSmoothing, trend=bestparams[0], damped=True)
                self.forecast_tests["HoltWinters"] = holtfc
                mape = np.mean(abs(holtfc["actual"] - holtfc["predict"])) / np.mean(holtfc["actual"]) * 100
                fluc = np.mean(holtfc["actual"].diff())
                maefluc = np.mean(abs(holtfc["actual"] - holtfc["predict"])) / fluc * 100
            else:
                holtfit = ExponentialSmoothing(self.series ,trend=bestparams[0]).fit()
                holtfc = self.onestep_forecast(self.series, fc_numtest, model=ExponentialSmoothing, trend=bestparams[0])
                self.forecast_tests["HoltWinters"] = holtfc
                mape = np.mean(abs(holtfc["actual"] - holtfc["predict"])) / np.mean(holtfc["actual"]) * 100
                fluc = np.mean(holtfc["actual"].diff())
                maefluc = np.mean(abs(holtfc["actual"] - holtfc["predict"])) / fluc * 100

        #check if residuals are iid
        residacf = self.check_residualacf(holtfit.resid, lags=resid_lags)
        residport = self.portmanteau(holtfit.resid, acf_nlags=resid_lags)

        if seasonal_periods:
            self.resid["Seasonal_HoltWinters"]=holtfit.resid
            self.residiid["Seasonal_HoltWinters"] = {"SampleACF": residacf, "Portmanteau": residport}
        else:
            self.resid["HoltWinters"]=holtfit.resid
            self.residiid["Seasonal_HoltWinters"] = {"SampleACF": residacf, "Portmanteau": residport}

        numpass = sum([residacf[0], residport[0], residport[2]])

        #set if 2 or more test shows NOT iid, indicate that residuals are not iid
        if numpass <= 1:
            #check for arch effect
            archind = self.engle_archtest(holtfit.resid)

            #save residual engle p-value as attribute
            if seasonal_periods:
                self.engle["Seasonal_HoltWinters"] = archind
            else:
                self.engle["HoltWinters"] = archind

            # if arch effect present, find best fit garch model, garch_search returns [best garch param, if standardised resid iid, if arch effect present in standardised resid]    
            if archind[0]:
                garch_res = self.garch_search(holtfit.resid, garch_dist=garch_dist)

                #save engle test p-value as attribute
                if seasonal_periods:
                    self.standardiid["Seasonal_HoltWinters"] = garch_res[1][1]
                    self.standard_engle["Seasonal_HoltWinters"] = garch_res[-1]
                else:
                    self.standardiid["HoltWinters"] = garch_res[1][1]
                    self.standard_engle["HoltWinters"] = garch_res[-1]

                res = [bestparams, holtfit.aic, numpass>=2, mape, maefluc, archind[0], garch_res[0], garch_res[1][0], garch_res[2][0]] 
                return res

            #event where residual not iid and have no arch effect
            else:
                return [bestparams, holtfit.aic, numpass>=2, mape, maefluc, archind[0], np.NaN, np.NaN, np.NaN]

        #event where holt has iid residuals
        else:
            return [bestparams, holtfit.aic, numpass>=2, mape, maefluc, np.NaN, np.NaN, np.NaN, np.NaN]

        

    def non_seasonal(self, window=60, fc_numtest=20, resid_lags=40, holtcheckdamped=True, garch_dist="t"):
        dfdict = {}
        tsdict = {0:self.series, 1:self.diff, 2:self.diff2}
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            #for arima difference 0,1,2
            for i in range(3):
                station_diff = self.test_stationarity(tsdict[i], disp=False, window=window)
                station_diff_ind = station_diff[0]

                #check if difference 0, 1 or 2 series is stationary
                if station_diff_ind:
                    if self.pbar:
                        print("ARIMA("+str(i)+") in progress")
                    arimares = self.arimagarch_search(difference=i, fc_numtest=fc_numtest, resid_lags=resid_lags, garch_dist=garch_dist)

                    # if conditions of iid standard residuals met and mape smaller than current, set model as best model attribute
                    if (arimares[2] and arimares[3]<self.bestmodel[-1]) or (arimares[2]==False and arimares[-2] and arimares[-1]==False and arimares[3]<self.bestmodel[-1]):
                        self.bestmodel = ["ARIMA("+str(i)+")", arimares[0], arimares[1], arimares[-3], arimares[3]]
                else:
                    arimares = ["Non-Stationary"] * 9
                dfdict["ARIMA("+str(i)+")"]=arimares

            if self.pbar:
                print("HoltWinters in progress")
            holtres = self.holtgarch(checkdamped=holtcheckdamped, fc_numtest=fc_numtest, resid_lags=resid_lags, garch_dist=garch_dist)
            # if conditions of iid standard residuals met and mape smaller than current, set model as best model attribute
            if (holtres[2] and holtres[3]<self.bestmodel[-1]) or (holtres[2]==False and holtres[-2] and holtres[-1]==False and holtres[3]<self.bestmodel[-1]):
                self.bestmodel = ["HoltWinters", holtres[0], holtres[1], holtres[-3], holtres[3]] 
            dfdict["HoltWinters"] = holtres

        newdf = pd.DataFrame(dfdict, index=["model_parameter", "model AIC", "residual IID", "Mean ABS % Error", "MAE/Fluctuation %", "residual ARCH effect", "GARCH_parameter",
                                            "standardised resid IID", "standardised resid ARCH"])
        self.nonseasonal_result = newdf
        return newdf

    def grid_SARIMA(self, ts , difference=1, interval=(0,3), seasonal_periods=[30,]):
        if difference != 1 or difference != 2:
            return "Only SARIMA of maximum difference 2 is accepted"
        
        # Define the p, d and q parameters to take any value between 0 and 3
        p = q = range(interval[0], interval[1])

        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, [difference], q))

        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = []
        for s in seasonal_periods:
            s_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(p, [1], q))]
            seasonal_pdq.extend(s_pdq)

        bestAIC = np.inf
        bestParam = None
        bestSParam = None
        
        dfdict = {}
        if self.pbar:
            from tqdm import tqdm
            itergrid = tqdm(pdq, desc="SARIMA Grid Search")
        else:
            itergrid = pdq
        
        #use gridsearch to look for optimial arima parameters
        for param in itergrid:
            for param_seasonal in seasonal_pdq:
                try:
                    
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        mod = SARIMAX(ts, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)

                        results = mod.fit()
                    
                    lab = (param, param_seasonal)
                    dfdict[lab] = []
                    rmse = sqrt(np.mean((series[1:] - results.fittedvalues[1:]) **2))
                    aic = results.aic
                    bic = results.bic
                    dfdict[lab].append(aic)
                    dfdict[lab].append(bic)
                    dfdict[lab].append(rmse)

                    #if current run of AIC is better than the best one so far, overwrite it
                    if results.aic<bestAIC:
                        bestAIC = results.aic
                        bestParam = param
                        bestSParam = param_seasonal

                except:
                    continue
        
        newdf = pd.DataFrame(dfdict, index=["aic", "bic", "RMSE Values"])
        sortdf = newdf.transpose().sort_values(by="aic")
        param = list(sortdf.index)
        sortdf["param"] = param
        sortdf = sortdf.reset_index(drop=True)
        
        return sortdf


    def sarimagarch_search(self, difference=1, seasonal_periods=[30,], grid_interval=(0,3), fc_numtest=20, resid_lags=40, garch_dist="t"):
        if difference == 1:
            ts = self.diff
        elif difference == 2:
            ts = self.diff2
        else:
            return "Only SARIMA of maximum difference 2 is accepted"

        gridsarima = self.grid_SARIMA(ts, difference=difference, interval=grid_interval, seasonal_periods=seasonal_periods)
        dlab = "SARIMA("+str(difference)+")"
        self.grids[dlab] = gridsarima

        bestparam = None
        bestsarifit = None

        for param in gridsarima["param"]:

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    model = SARIMAX(ts, order=param[0], seasonal_order=param[1], enforce_stationarity=False, enforce_invertibility=False)
                    sarimafit = model.fit()

                    bestsarifit = sarimafit
                    bestparam = param
                    break
            except:
                continue

        if bestparam:
            sarifc = self.onestep_forecast(self.series, fc_numtest, model=SARIMAX, order=bestparam[0], seasonal_order=bestparam[1], enforce_stationarity=False, enforce_invertibility=False)
            self.forecast_tests[dlab] = sarifc
            mape = np.mean(abs(sarifc["actual"] - sarifc["predict"])) / np.mean(sarifc["actual"]) * 100
            fluc = np.mean(sarifc["actual"].diff())
            maefluc = np.mean(abs(sarifc["actual"] - sarifc["predict"])) / fluc * 100

            self.resid[dlab] = bestsarifit.resid
            #check if residuals are iid
            residacf = self.check_residualacf(bestsarifit.resid, lags=resid_lags)
            residport = self.portmanteau(bestsarifit.resid, acf_nlags=resid_lags)
            self.residiid[dlab] = {"SampleACF": residacf, "Portmanteau": residport}

            numpass = sum([residacf[0], residport[0], residport[2]])

            #set if 2 or more test shows NOT iid, indicate that residuals are not iid
            if numpass <= 1:
                #check for arch effect
                archind = self.engle_archtest(bestsarifit.resid)
                self.engle[dlab] = archind
                    
                # if arch effect present, find best fit garch model, garch_search returns [best garch param, if standardised resid iid, if arch effect present in standardised resid]    
                if archind[0]:

                    garch_res = self.garch_search(bestsarifit.resid, garch_dist=garch_dist)
                    self.standardiid[dlab] = garch_res[1][1]
                    self.standard_engle[dlab] = garch_res[-1]

                    res = [bestparam, bestsarifit.aic, numpass>=2, mape, maefluc, archind[0], garch_res[0], garch_res[1][0], garch_res[2][0]] 
                    return res

                #event where residual not iid and have no arch effect
                else:
                    return [bestparam, bestsarifit.aic, numpass>=2, mape, maefluc, archind[0], np.NaN, np.NaN, np.NaN]

            #event where sarima has iid residuals
            else:
                return [bestparam, bestsarifit.aic, numpass>=2, mape, maefluc, np.NaN, np.NaN, np.NaN, np.NaN]

        #event where no parameters fit sarima model
        else:
            return [np.NaN] * 9

    def seasonal(self, seasonal_periods=[30,], diff2=False, grid_interval=(0,3), window=60, fc_numtest=20, resid_lags=40, holtcheckdamped=False, garch_dist="t"):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            station_diff = self.test_stationarity(self.diff, disp=False, window=window)
            station_diff_ind = station_diff[0]

            #check if difference 1 series is stationary
            if station_diff_ind:
                if self.pbar:
                    print("SARIMA(1) in progress")
                sarima1res = self.sarimagarch_search(difference=1, seasonal_periods=seasonal_periods, grid_interval=grid_interval, fc_numtest=fc_numtest, resid_lags=resid_lags, garch_dist=garch_dist)
                # if conditions of iid standard residuals met and mape smaller than current, set model as best model attribute
                if (sarima1res[2] and sarima1res[3]<self.bestmodel[-1]) or (sarima1res[2]==False and sarima1res[-2] and sarima1res[-1]==False and sarima1res[3]<self.bestmodel[-1]):
                    self.bestmodel = ["SARIMA(1)", sarima1res[0], sarima1res[1], sarima1res[-3], sarima1res[3]]
            else:
                sarima1res = ["Non-Stationary"] * 9

            if diff2:
                station_diff2 = self.test_stationarity(self.diff2, disp=False, window=window)
                station_diff2_ind = station_diff2[0]

                #check if difference 2 series is stationary
                if station_diff_ind:
                    if self.pbar:
                        print("SARIMA(2) in progress")
                    sarima2res = self.sarimagarch_search(difference=2, seasonal_periods=seasonal_periods, grid_interval=grid_interval, fc_numtest=fc_numtest, resid_lags=resid_lags, garch_dist=garch_dist)
                    # if conditions of iid standard residuals met and mape smaller than current, set model as best model attribute
                    if (sarima2res[2] and sarima2res[3]<self.bestmodel[-1]) or (sarima2res[2]==False and sarima2res[-2] and sarima2res[-1]==False and sarima2res[3]<self.bestmodel[-1]):
                        self.bestmodel = ["SARIMA(2)", sarima2res[0], sarima2res[1], sarima2res[-3], sarima2res[3]]
                else:
                    sarima2res = ["Non-Stationary"] * 9

            if self.pbar:
                print("HoltWinters in progress")
            holtres = self.holtgarch(checkdamped=holtcheckdamped, seasonal_periods=seasonal_periods, fc_numtest=fc_numtest, resid_lags=resid_lags, garch_dist=garch_dist)
            # if conditions of iid standard residuals met and mape smaller than current, set model as best model attribute
            if (holtres[2] and holtres[3]<self.bestmodel[-1]) or (holtres[2]==False and holtres[-2] and holtres[-1]==False and holtres[3]<self.bestmodel[-1]):
                self.bestmodel = ["Seasonal_HoltWinters", holtres[0], holtres[1], holtres[-3], holtres[3]]

        if diff2:
            dfdict = {"SARIMA(1)": sarima1res, "SARIMA(2)": sarima2res, "Seasonal HoltWinters": holtres}
        else:
            dfdict = {"SARIMA(1)": sarima1res, "Seasonal HoltWinters": holtres}

        newdf = pd.DataFrame(dfdict, index=["model_parameter", "model AIC", "residual IID", "Mean ABS % Error", "MAE/Fluctuation %", "residual ARCH effect", "GARCH_parameter",
                                                "standardised resid IID", "standardised resid ARCH"])
        self.seasonal_result = newdf
        return newdf

    def onestep_CIfc(self, ts, totalnum_test=20, garch_order=None, garch_dist="t", model=ARIMA, **kwargs):
        if model==ARIMA:
            if garch_order:
                df_dict = {"actual":[], "predict":[], "difference":[], "ari_lower":[], "ari_upper":[], "garch_lower":[], "garch_upper":[], "date":[]}
            else:
                df_dict = {"actual":[], "predict":[], "difference":[], "ari_lower":[], "ari_upper":[]}
        else:
            df_dict = {"actual":[], "predict":[], "difference":[], "garch_lower":[], "garch_upper":[], "date":[]}

        ts = ts.sort_index()
        if self.pbar:
            from tqdm import tqdm
            iterfc = tqdm(range(totalnum_test), desc="Forecast CI Test")
        else:
            iterfc = range(totalnum_test)
        for i in iterfc:
            index = totalnum_test - i
            train = ts[:-index]
            actual = ts[-index]
            date = ts.index[-index]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                inimodel = model(train, **kwargs)
                
                if model == ARIMA:
                    fcfit = inimodel.fit()
                    predictdata = fcfit.forecast(1)
                    predictval = predictdata[0]
                else:
                    fcfit = inimodel.fit()
                    predictval = np.array(fcfit.forecast(1))

                if garch_order:
                    history = fcfit.resid
                    garchmodel = arch_model(history, mean="constant", p=garch_order[0], q=garch_order[1], dist='normal')
                    res = garchmodel.fit(update_freq=5, disp='off')
                    
                    pred_var = res.forecast(horizon=1, start=None, align='origin').variance["h.1"].iloc[-1]
                    pred_sd = sqrt(pred_var)
                    pred_mean = res.forecast(horizon=1, start=None, align='origin').mean["h.1"].iloc[-1]
                
            #predict = fcfit.forecast(1)[0]
            diff = actual - predictval[0]
            
            if model==ARIMA:
                ari95 = predictdata[2][0]

            if garch_order:
                if garch_dist == "t":
                    garch95 = st.t.interval(df=len(history)-1, loc=pred_mean, scale=pred_sd, alpha=0.95)
                elif garch_dist == "normal":
                    garch95 = st.norm.interval(loc=pred_mean, scale=pred_sd, alpha=0.95)
                else:
                    return "Only \"t\" and \"normal\" accepted for garch_dist parameter."
                    
            df_dict["date"].append(date)

            if garch_order:
                df_dict["garch_upper"].append(garch95[1] + predictval[0])
                df_dict["garch_lower"].append(garch95[0] + predictval[0])
            
            if model==ARIMA:
                df_dict["ari_upper"].append(ari95[1])
                df_dict["ari_lower"].append(ari95[0])
            
            df_dict["difference"].append(diff)
            df_dict["predict"].append(predictval[0])
            df_dict["actual"].append(actual)
        newdf = pd.DataFrame(df_dict)
        newdf = newdf.set_index(["date"])
        return newdf

    def analyse_fit(self, actual, fitted, fig_dim=(14,6), day_interval=40, cond_interval=None, labels=["ARIMA", "GARCH"]):
        rms = sqrt(np.mean((actual - fitted) **2))
        rmsp = rms / np.mean(actual) * 100
        
        mape = np.mean(abs(actual - fitted)) / np.mean(actual) * 100
        fluc = np.mean(actual.diff())
        print("Root Mean Square Error:", rms)
        print("RMSE / Mean (%):", rmsp, "%")
        print("Mean Absolute Percentage Error (%):", mape, "%")
        print("Mean Fluctuation each day:", fluc)
        print("MAE / Mean Fluctuation (%):", np.mean(abs(actual - fitted))/fluc * 100, "%")

        plt.figure(figsize=fig_dim)
        plt.plot(actual.index, actual.values, color="blue", label="actual")
        plt.plot(fitted.index, fitted.values, color="red", label="fitted")
        if cond_interval:
            if len(cond_interval)==4:
                plt.fill_between(cond_interval[0].index, cond_interval[0].values, cond_interval[1].values, label=labels[0]+" Prediction Interval", color="gold", alpha=0.5)
                plt.fill_between(cond_interval[2].index, cond_interval[2].values, cond_interval[3].values, label=labels[1]+" Prediction Interval", color="lightblue", alpha=0.3)
            elif len(cond_interval)==2:
                plt.fill_between(cond_interval[0].index, cond_interval[0].values, cond_interval[1].values, label=labels[1]+" Prediction Interval", color="lightblue", alpha=0.3)       
            
        plt.legend()
        plt.gca().xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=day_interval)) 
        plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()
        plt.show()

    def analyse_fit_auto(self, df, day_interval=40):
        num_col = len(df.columns)

        if num_col==3:
            return self.analyse_fit(df["actual"], df["predict"], day_interval=day_interval)
        elif num_col==5:
            if "garch_upper" in list(df.columns):
                return self.analyse_fit(df["actual"], df["predict"], day_interval=day_interval, cond_interval=[df["garch_lower"], df["garch_upper"]])
            elif "ari_upper" in list(df.columns):
                return self.analyse_fit(df["actual"], df["predict"], day_interval=day_interval, cond_interval=[df["ari_lower"], df["ari_upper"]], labels=["ARIMA", "ARIMA"])
            else:
                return "No Auto fitting plot found"
        elif num_col==7:
            return self.analyse_fit(df["actual"], df["predict"], day_interval=day_interval, cond_interval=[df["ari_lower"], df["ari_upper"], df["garch_lower"], df["garch_upper"]])
        else:
            return "No Auto fitting plot found"

    def get_best_distribution(self, data, test_size=20):
        #all scipy distribution except levy_stable  and kstwobign as fitting on levy_stable and kstwobign raise an error
        distribution = [       
            st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
            st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
            st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
            st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
            st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
            st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.laplace,st.levy,st.levy_l,
            st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
            st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
            st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
            st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
        ]
       
        dist_results = []
        params = {}
        for dist in distribution:
           
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                   
                    dist_name = dist.name
                    param = dist.fit(data)
                    params[dist_name] = param
                    # Applying the Kolmogorov-Smirnov test
                    D, p = st.kstest(data, dist_name, args=param, N=test_size)
                    #print("p value for "+dist_name+" = "+str(p))
                    dist_results.append((dist_name, p, dist))
            except Exception:
                pass
        # select the best fitted distribution
        #print("number of p values:"+str(len(dist_results)))
        best_distname, best_p, best_dist = (max(dist_results, key=lambda item: item[1]))
        # store the name of the best fit and its p value
        return best_distname, best_p, params[best_distname], best_dist

    #For Holt-Winters only GARCH prediction interval supported
    def predict_next(self, detail=False, garch_order=None, garch_dist="t", alpha=0.95, model=ARIMA, **kwargs):
        ts = self.series

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
                
            inimodel = model(ts, **kwargs)
                
            if model == ARIMA:
                fcfit = inimodel.fit()
                arialpha = 1-alpha
                predictdata = fcfit.forecast(1, alpha=arialpha)
                predictval = predictdata[0]
            else:
                fcfit = inimodel.fit()
                predictval = np.array(fcfit.forecast(1))

            predictionary = {"forecast": predictval[0]}

            if garch_order:
                history = fcfit.resid
                garchmodel = arch_model(history, mean="constant", p=garch_order[0], q=garch_order[1], dist='normal')
                res = garchmodel.fit(update_freq=5, disp='off')
                    
                pred_var = res.forecast(horizon=1, start=None, align='origin').variance["h.1"].iloc[-1]
                pred_sd = sqrt(pred_var)
                pred_mean = res.forecast(horizon=1, start=None, align='origin').mean["h.1"].iloc[-1]
            else:
                if model==ExponentialSmoothing or model==SARIMAX:
                    history = fcfit.resid
                    dist_name, p, params, dist = self.get_best_distribution(history)
                    fitint = dist.interval(alpha, *params)
                    predictionary["prediction_interval"] = [fitint[0] + predictval[0], fitint[1] + predictval[0]]

        if garch_order:
            if model==ARIMA and detail:
                ari_int = predictdata[2][0]
                predictionary["ARIMA_interval"] = list(ari_int)
            if garch_dist == "t":
                garchint = st.t.interval(df=len(history)-1, loc=pred_mean, scale=pred_sd, alpha=alpha)
                if detail:
                    predictionary["GARCH_interval"] = [garchint[0] + predictval[0], garchint[1] + predictval[0]]
                else:
                    predictionary["prediction_interval"] = [garchint[0] + predictval[0], garchint[1] + predictval[0]]
            elif garch_dist == "normal":
                garchint = st.norm.interval(loc=pred_mean, scale=pred_sd, alpha=alpha)
                if detail:
                    predictionary["GARCH_interval"] = [garchint[0] + predictval[0], garchint[1] + predictval[0]]
                else:
                    predictionary["prediction_interval"] = [garchint[0] + predictval[0], garchint[1] + predictval[0]]
            else:
                return "Only \"t\" and \"normal\" accepted for garch_dist parameter."
        else:
            if model==ARIMA:
                ari_int = predictdata[2][0]
                if detail:
                    predictionary["ARIMA_interval"] = list(ari_int)
                else:
                    predictionary["prediction_interval"] = list(ari_int)
        return predictionary

    def predict_with_bestmodel(self, detail=False, garch_dist="t", alpha=0.95):
        if self.bestmodel == [np.NaN, np.NaN, np.NaN, np.NaN, np.Inf]:
            return "There is currently no bestmodel to predict from. Run non_seasonal() or seasonal() methods to generate models"
        elif "ARIMA" in self.bestmodel[0] and "SARIMA" not in self.bestmodel[0]:
            if type(self.bestmodel[3])!=tuple and np.isnan(self.bestmodel[3]):
                return self.predict_next(detail=detail, alpha=alpha, order=self.bestmodel[1])
            else:
                return self.predict_next(detail=detail, garch_order=self.bestmodel[3], garch_dist=garch_dist, alpha=alpha, order=self.bestmodel[1])
        elif self.bestmodel[0]=="SARIMA(1)" or self.bestmodel[0]=="SARIMA(2)":
            if type(self.bestmodel[3])!=tuple and np.isnan(self.bestmodel[3]):
                return self.predict_next(detail=detail, alpha=alpha, order=self.bestmodel[1][0], seasonal_order=self.bestmodel[1][1], enforce_stationarity=False, enforce_invertibility=False)
            else:
                return self.predict_next(detail=detail, garch_order=self.bestmodel[3], garch_dist=garch_dist, alpha=alpha, model=SARIMAX,
                                         order=self.bestmodel[1][0], seasonal_order=self.bestmodel[1][1], enforce_stationarity=False, enforce_invertibility=False)
        elif self.bestmodel[0]=="Seasonal_HoltWinters":
            if type(self.bestmodel[3])!=tuple and np.isnan(self.bestmodel[3]):
                if len(trend=bestmodel[1]) == 2:
                    return self.predict_next(detail=detail, alpha=alpha, model=ExponentialSmoothing, trend=self.bestmodel[1][0][0], damped=True, seasonal=self.bestmodel[1][1][0],
                                             seasonal_periods=self.bestmodel[1][1][1])
                else:
                    return self.predict_next(detail=detail, alpha=alpha, model=ExponentialSmoothing, trend=self.bestmodel[1][0][0], seasonal=self.bestmodel[1][1][0], seasonal_periods=self.bestmodel[1][1][1])
            else:
                if len(trend=bestmodel[1]) == 2:
                    return self.predict_next(detail=detail, garch_order=self.bestmodel[3], garch_dist=garch_dist, alpha=alpha, model=ExponentialSmoothing, trend=self.bestmodel[1][0][0],
                                             damped=True, seasonal=self.bestmodel[1][1][0], seasonal_periods=self.bestmodel[1][1][1])
                else:
                    return self.predict_next(detail=detail, garch_order=self.bestmodel[3], garch_dist=garch_dist, alpha=alpha, model=ExponentialSmoothing, trend=self.bestmodel[1][0][0],
                                             seasonal=self.bestmodel[1][1][0], seasonal_periods=self.bestmodel[1][1][1])
        elif self.bestmodel[0]=="HoltWinters":
            if type(self.bestmodel[3])!=tuple and np.isnan(self.bestmodel[3]):
                if len(trend=bestmodel[1]) == 2:
                    return self.predict_next(detail=detail, alpha=alpha, model=ExponentialSmoothing, trend=self.bestmodel[1][0][0], damped=True)
                else:
                    return self.predict_next(detail=detail, alpha=alpha, model=ExponentialSmoothing, trend=self.bestmodel[1][0][0])
            else:
                if len(trend=bestmodel[1]) == 2:
                    return self.predict_next(detail=detail, garch_order=self.bestmodel[3], garch_dist=garch_dist, alpha=alpha, model=ExponentialSmoothing, trend=self.bestmodel[1][0][0],
                                             damped=True)
                else:
                    return self.predict_next(detail=detail, garch_order=self.bestmodel[3], garch_dist=garch_dist, alpha=alpha, model=ExponentialSmoothing, trend=self.bestmodel[1][0][0])
        else:
            return "Model not recognised. Please use predict_next() method instead"

    def predict_next_auto(self, detail=False, seasonal=False, disp=False, garch_dist="t", alpha=0.95, **kwargs):
        if seasonal:
            search_res = self.seasonal(**kwargs)
        else:
            search_res = self.non_seasonal(**kwargs)

        if disp:
            print(search_res)

        return self.predict_with_bestmodel(detail=detail, garch_dist=garch_dist, alpha=alpha)
            
#author: Jerald Teo 2019
