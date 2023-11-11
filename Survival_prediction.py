{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model for Survival Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import sys\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "\n",
    "from lifelines import KaplanMeierFitter\n",
    "from sksurv.util import Surv\n",
    "from sksurv.ensemble import RandomSurvivalForest\n",
    "from sksurv.metrics import brier_score\n",
    "from sksurv.metrics import concordance_index_censored\n",
    "from sksurv.metrics import concordance_index_ipcw\n",
    "from sksurv.metrics import cumulative_dynamic_auc\n",
    "from sksurv.metrics import integrated_brier_score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 0. Loading dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "df = pd.read_excel('/Nas/0_All_rev.xlsx', sheet_name='All', engine='openpyxl')\n",
    "\n",
    "df = df.fillna(0)\n",
    "for i in df.columns[1:-1]:\n",
    "    for j in range(len(df)):\n",
    "        df.loc[j,i] = float(df.loc[j,i])\n",
    "        \n",
    "df_KU = df[df['Center'] == 1]\n",
    "df_BH = df[df['Center'] == 2]\n",
    "df_SM = df[df['Center'] == 3]\n",
    "df_SN = df[df['Center'] == 4]\n",
    "df_CM = df[df['Center'] == 5]\n",
    "df_SV = df[df['Center'] == 6]\n",
    "df_AM = df[df['Center'] == 7]\n",
    "df_CA = df[df['Center'] == 8]\n",
    "df_IH = df[df['Center'] == 9]\n",
    "df_ex = df[df['Center'] != 7]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Model training and validation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "### 1.1. Internal validation with 5-fold cross validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "X = df_AM.loc[:, 'age':'Tx']\n",
    "y = Surv.from_dataframe('death_01', 'death_mo', df_AM)\n",
    "skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 999)\n",
    "rsf = RandomSurvivalForest(random_state=999)\n",
    "\n",
    "df = pd.DataFrame()\n",
    "for i, (train_index, test_index) in enumerate(skf.split(X, y)):\n",
    "\n",
    "    X_train = X.loc[train_index, :]\n",
    "    y_train = y[train_index]\n",
    "    X_test = X.loc[test_index, :]\n",
    "    y_test = y[test_index]\n",
    "\n",
    "    # model training for internal dataset with 5-fold CV\n",
    "    estimator = rsf.fit(X_train, y_train)\n",
    "    \n",
    "    event_min = y_train[y_train[\"death_01\"] == 1][\"death_mo\"].min()\n",
    "    event_max = y_train[y_train[\"death_01\"] == 1][\"death_mo\"].max()\n",
    "    idx = np.where((event_min <= y_test[\"death_mo\"]) & (y_test[\"death_mo\"] < event_max))\n",
    "    idx = np.array(list(idx)).flatten()\n",
    "\n",
    "    X_test = X_test.loc[X_test.index[idx], :]\n",
    "    y_test = y_test[idx]\n",
    "\n",
    "    # model testing for internal dataset\n",
    "    chf_funcs = estimator.predict_cumulative_hazard_function(X_test)\n",
    "    surv_funcs = estimator.predict_survival_function(X_test)\n",
    "    times = np.percentile(y_test[\"death_mo\"], np.linspace(5, 81, 15))\n",
    "    risk_scores = np.row_stack([chf(times) for chf in chf_funcs]) \n",
    "\n",
    "    c_index = estimator.score(X_test, y_test)\n",
    "    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times) \n",
    "    preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])\n",
    "    IBS = integrated_brier_score(y_train, y_test, preds, times)    \n",
    "\n",
    "    df.loc[i, 'C_index'] = c_index\n",
    "    df.loc[i, 'Mean_auc'] = mean_auc\n",
    "    df.loc[i, 'IBS'] = IBS\n",
    "    \n",
    "for col in df.columns:\n",
    "    df.loc['Mean', col] = df[col].mean()\n",
    "    df.loc['SD', col] = df[col].std()\n",
    "    \n",
    "internal = df.copy()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.2. External validation after training with whole internal dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# model training with whole internal dataset\n",
    "X = df_AM.loc[:, 'age':'Tx']\n",
    "y = Surv.from_dataframe('death_01', 'death_mo', df_AM)\n",
    "estimator = rsf.fit(X, y)\n",
    "\n",
    "# model testing for external validation datasets \n",
    "centers = [\"KU\", \"BH\", \"SM\", \"SN\", \"CM\", \"SV\", \"CA\", \"IH\"]\n",
    "mod = sys.modules[__name__]\n",
    "\n",
    "def external_validaiton(model, y_train):\n",
    "    \n",
    "    df = pd.DataFrame()\n",
    "\n",
    "    for i, center in enumerate(centers):\n",
    "        test_data = getattr(mod, f\"df_{center}\")\n",
    "        \n",
    "        test_data.reset_index(drop=True, inplace=True)\n",
    "        \n",
    "        X_test = test_data.loc[:, 'age':'Tx']\n",
    "        y_test = Surv.from_dataframe('death_01', 'death_mo', test_data)\n",
    "\n",
    "        event_min = y_train[y_train[\"death_01\"] == 1][\"death_mo\"].min()\n",
    "        event_max = y_train[y_train[\"death_01\"] == 1][\"death_mo\"].max()\n",
    "        idx = np.where((event_min <= y_test[\"death_mo\"]) & (y_test[\"death_mo\"] < event_max))\n",
    "        idx = np.array(list(idx)).flatten()\n",
    "\n",
    "        X_test = X_test.loc[X_test.index[idx], :]\n",
    "        y_test = y_test[idx]\n",
    "\n",
    "        chf_funcs = estimator.predict_cumulative_hazard_function(X_test)\n",
    "        surv_funcs = estimator.predict_survival_function(X_test)\n",
    "        times = np.percentile(y_test[\"death_mo\"], np.linspace(5, 81, 15))\n",
    "        risk_scores = np.row_stack([chf(times) for chf in chf_funcs]) \n",
    "        \n",
    "        c_index = estimator.score(X_test, y_test)\n",
    "        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times) \n",
    "        preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])\n",
    "        IBS = integrated_brier_score(y_train, y_test, preds, times)    \n",
    "\n",
    "        df.loc[i, 'Center'] = center\n",
    "        df.loc[i, 'C_index'] = c_index\n",
    "        df.loc[i, 'Mean_auc'] = mean_auc\n",
    "        df.loc[i, 'IBS'] = IBS\n",
    "    \n",
    "    return (df)\n",
    "\n",
    "# external validation results\n",
    "df = external_validaiton(estimator, y)\n",
    "\n",
    "for col in df.columns[1:]:\n",
    "    df.loc['Mean', col] = df[col].mean()\n",
    "    df.loc['SD', col] = df[col].std()\n",
    "    \n",
    "external = df.copy()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.3. Training and validation with individual dataset "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "df_all = pd.DataFrame()\n",
    "\n",
    "for n, center in enumerate(centers):\n",
    "    \n",
    "    test_data = getattr(mod, f\"df_{center}\")\n",
    "    test_data.reset_index(drop=True, inplace=True)\n",
    "    \n",
    "    X = test_data.loc[:, 'age':'Tx']\n",
    "    Tx = test_data.loc[:, 'Tx']\n",
    "    y = Surv.from_dataframe('death_01', 'death_mo', test_data)\n",
    "    df = pd.DataFrame()\n",
    "    \n",
    "    # model training with individual dataset with 5-fold CV\n",
    "    for i, (train_index, test_index) in enumerate(skf.split(X, y)):\n",
    "        X_train = X.loc[train_index, :]\n",
    "        y_train = y[train_index]\n",
    "        X_test = X.loc[test_index, :]\n",
    "        y_test = y[test_index]\n",
    "        \n",
    "        estimator = rsf.fit(X_train, y_train)\n",
    "\n",
    "        event_min = y_train[y_train[\"death_01\"] == 1][\"death_mo\"].min()\n",
    "        event_max = y_train[y_train[\"death_01\"] == 1][\"death_mo\"].max()\n",
    "        idx = np.where((event_min <= y_test[\"death_mo\"]) & (y_test[\"death_mo\"] < event_max))\n",
    "        idx = np.array(list(idx)).flatten()\n",
    "        X_test = X_test.loc[X_test.index[idx], :]\n",
    "        y_test = y_test[idx]\n",
    "        \n",
    "        # model testing for individual dataset\n",
    "        chf_funcs = estimator.predict_cumulative_hazard_function(X_test)\n",
    "        surv_funcs = estimator.predict_survival_function(X_test)\n",
    "        times = np.percentile(y_test[\"death_mo\"], np.linspace(5, 81, 15))\n",
    "        risk_scores = np.row_stack([chf(times) for chf in chf_funcs]) \n",
    "\n",
    "        c_index = estimator.score(X_test, y_test)\n",
    "        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times) \n",
    "        preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])\n",
    "        IBS = integrated_brier_score(y_train, y_test, preds, times)    \n",
    "\n",
    "        df.loc[i, 'C_index'] = c_index\n",
    "        df.loc[i, 'Mean_auc'] = mean_auc\n",
    "        df.loc[i, 'IBS'] = IBS\n",
    "        \n",
    "    for col in df.columns:\n",
    "        df.loc['Mean', col] = df[col].mean()\n",
    "        df.loc['SD', col] = df[col].std()\n",
    "        \n",
    "    df_all = pd.concat([df_all, df])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Model training and validation for each specific treatment"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "source": [
    "### 3.1. Internal validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "Tx = df_AM.loc[:, 'Tx']\n",
    "\n",
    "X.reset_index(drop=True, inplace=True)\n",
    "Tx_name = ['RFA', 'Op', 'TACE', 'TACE+RT', 'Sorafenib', 'None']\n",
    "df = pd.DataFrame()\n",
    "\n",
    "# 5-fold CV stratified by treatment\n",
    "for i, (train_index, test_index) in enumerate(skf.split(X, Tx)):\n",
    "\n",
    "    X_train_fold = X.loc[train_index, :]\n",
    "    y_train_fold = y[train_index]\n",
    "    X_test_fold = X.loc[test_index, :]\n",
    "    y_test_fold = y[test_index]\n",
    "\n",
    "    # composing dataset including only specific treatment\n",
    "    for tx, tx_name in enumerate(Tx_name):\n",
    "        train_idx = X_train_fold[X_train_fold['Tx'] == tx].index\n",
    "        test_idx = X_test_fold[X_test_fold['Tx'] == tx].index\n",
    "        \n",
    "        X_train = X.loc[train_idx, :]\n",
    "        y_train = y[train_idx]\n",
    "        X_test = X.loc[test_idx, :]\n",
    "        y_test = y[test_idx]\n",
    "        \n",
    "        # model training with training dataset including only specific treatment\n",
    "        estimator = rsf.fit(X_train, y_train)\n",
    "        \n",
    "        if len(X_test) == 0:\n",
    "            df.loc[i, f'{tx}_Tx_name'] = tx_name\n",
    "            df.loc[i, f'{tx}_C_index'] = 0\n",
    "            df.loc[i, f'{tx}_Mean_auc'] = 0\n",
    "            df.loc[i, f'{tx}_IBS'] = 0\n",
    "                \n",
    "        else:\n",
    "            event_min = y_train[y_train[\"death_01\"] == 1][\"death_mo\"].min()\n",
    "            event_max = y_train[y_train[\"death_01\"] == 1][\"death_mo\"].max()\n",
    "            idx = np.where((event_min <= y_test[\"death_mo\"]) & (y_test[\"death_mo\"] < event_max))\n",
    "            idx = np.array(list(idx)).flatten()\n",
    "\n",
    "            X_test = X_test.loc[X_test.index[idx], :]\n",
    "            y_test = y_test[idx]\n",
    "\n",
    "            # model testing with training dataset including only specific treatment\n",
    "            chf_funcs = estimator.predict_cumulative_hazard_function(X_test)\n",
    "            surv_funcs = estimator.predict_survival_function(X_test)\n",
    "            times = np.percentile(y_test[\"death_mo\"], np.linspace(5, 81, 10))\n",
    "            times = np.unique(times)\n",
    "            risk_scores = np.row_stack([chf(times) for chf in chf_funcs]) \n",
    "\n",
    "            c_index = estimator.score(X_test, y_test)\n",
    "            auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times) \n",
    "            preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])\n",
    "            IBS = integrated_brier_score(y_train, y_test, preds, times)    \n",
    "\n",
    "            df.loc[i, f'{tx}_Tx_name'] = tx_name\n",
    "            df.loc[i, f'{tx}_C_index'] = c_index\n",
    "            df.loc[i, f'{tx}_Mean_auc'] = mean_auc\n",
    "            df.loc[i, f'{tx}_IBS'] = IBS  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "source": [
    "### 3.2. External validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "y_train = y.copy()\n",
    "df = pd.DataFrame()\n",
    "\n",
    "for i, center in enumerate(centers):\n",
    "    print (i)\n",
    "    \n",
    "    test_data = getattr(mod, f\"df_{center}\")\n",
    "    test_data.reset_index(drop=True, inplace=True)\n",
    "\n",
    "    X_test = test_data.loc[:, 'age':'Tx']\n",
    "    y_test = Surv.from_dataframe('death_01', 'death_mo', test_data)\n",
    "\n",
    "    for tx, tx_name in enumerate(Tx_name):\n",
    "        \n",
    "        train_idx = X[X['Tx'] == tx].index\n",
    "        X_train = X.loc[train_idx, :]\n",
    "        y_train = y[train_idx]\n",
    "        estimator = rsf.fit(X_train, y_train)\n",
    "\n",
    "        test_idx = X_test[X_test['Tx'] == tx].index\n",
    "        new_X_test = X.loc[test_idx, :]\n",
    "        new_y_test = y[test_idx]\n",
    "        \n",
    "        event_min = y_train[y_train[\"death_01\"] == 1][\"death_mo\"].min()\n",
    "        event_max = y_train[y_train[\"death_01\"] == 1][\"death_mo\"].max()\n",
    "        idx = np.where((event_min <= new_y_test[\"death_mo\"]) & (new_y_test[\"death_mo\"] < event_max))\n",
    "        idx = np.array(list(idx)).flatten()\n",
    "\n",
    "        new_X_test = new_X_test.loc[new_X_test.index[idx], :]\n",
    "        new_y_test = new_y_test[idx]\n",
    "        \n",
    "        if len(new_X_test) < 2:\n",
    "            df.loc[i, f'{tx}_Tx_name'] = tx_name\n",
    "            df.loc[i, f'{tx}_Test_no'] = len(new_X_test)\n",
    "            df.loc[i, f'{tx}_C_index'] = 0\n",
    "            df.loc[i, f'{tx}_Mean_auc'] = 0\n",
    "            df.loc[i, f'{tx}_IBS'] = 0\n",
    "            \n",
    "        else:\n",
    "            chf_funcs = estimator.predict_cumulative_hazard_function(new_X_test)\n",
    "            surv_funcs = estimator.predict_survival_function(new_X_test)\n",
    "            times = np.percentile(np.unique(new_y_test[\"death_mo\"]), np.linspace(5, 81, 10))\n",
    "            times = np.unique(times)\n",
    "            risk_scores = np.row_stack([chf(times) for chf in chf_funcs]) \n",
    "\n",
    "            c_index = estimator.score(new_X_test, new_y_test)\n",
    "            auc, mean_auc = cumulative_dynamic_auc(y_train, new_y_test, risk_scores, times) \n",
    "            preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])\n",
    "            IBS = integrated_brier_score(y_train, new_y_test, preds, times)    \n",
    "\n",
    "            df.loc[i, f'{tx}_Tx_name'] = tx_name\n",
    "            df.loc[i, f'{tx}_Test_no'] = len(new_X_test)\n",
    "            df.loc[i, f'{tx}_C_index'] = c_index\n",
    "            df.loc[i, f'{tx}_Mean_auc'] = mean_auc\n",
    "            df.loc[i, f'{tx}_IBS'] = IBS"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "source": [
    "### 3.3. Individual training & validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "jupyter": {
     "source_hidden": true
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df_all = pd.DataFrame()\n",
    "\n",
    "for n, center in enumerate(centers):\n",
    "    \n",
    "    print (n)\n",
    "    test_data = getattr(mod, f\"df_{center}\")\n",
    "    test_data.reset_index(drop=True, inplace=True)\n",
    "    \n",
    "    X = test_data.loc[:, 'age':'Tx']\n",
    "    Tx = test_data.loc[:, 'Tx']\n",
    "    y = Surv.from_dataframe('death_01', 'death_mo', test_data)\n",
    "    df = pd.DataFrame()\n",
    "    \n",
    "    for i, (train_index, test_index) in enumerate(skf.split(X, Tx)):\n",
    "\n",
    "        X_train = X.loc[train_index, :]\n",
    "        y_train = y[train_index]\n",
    "        X_test = X.loc[test_index, :]\n",
    "        y_test = y[test_index]\n",
    "        \n",
    "        estimator = rsf.fit(X_train, y_train)\n",
    "\n",
    "        for tx, tx_name in enumerate(Tx_name):\n",
    "            tmp_idx = X_test[X_test['Tx'] == tx].index\n",
    "            new_X_test = X.loc[tmp_idx, :]\n",
    "            new_y_test = y[tmp_idx]\n",
    "\n",
    "            event_min = y_train[y_train[\"death_01\"] == 1][\"death_mo\"].min()\n",
    "            event_max = y_train[y_train[\"death_01\"] == 1][\"death_mo\"].max()\n",
    "            idx = np.where((event_min <= new_y_test[\"death_mo\"]) & (new_y_test[\"death_mo\"] < event_max))\n",
    "            idx = np.array(list(idx)).flatten()\n",
    "\n",
    "            new_X_test = new_X_test.loc[new_X_test.index[idx], :]\n",
    "            new_y_test = new_y_test[idx]\n",
    "\n",
    "            if len(new_X_test) < 2:\n",
    "                df.loc[i, f'{tx}_Tx_name'] = tx_name\n",
    "                df.loc[i, f'{tx}_Test_no'] = len(new_X_test)\n",
    "                df.loc[i, f'{tx}_C_index'] = 0\n",
    "                df.loc[i, f'{tx}_Mean_auc'] = 0\n",
    "                df.loc[i, f'{tx}_IBS'] = 0\n",
    "            \n",
    "            else:\n",
    "                chf_funcs = estimator.predict_cumulative_hazard_function(new_X_test)\n",
    "                surv_funcs = estimator.predict_survival_function(new_X_test)\n",
    "                times = np.percentile(new_y_test[\"death_mo\"], np.linspace(5, 81, 10))\n",
    "                times = np.unique(times)\n",
    "                risk_scores = np.row_stack([chf(times) for chf in chf_funcs]) \n",
    "\n",
    "                try:\n",
    "                    c_index = estimator.score(new_X_test, new_y_test)\n",
    "                    auc, mean_auc = cumulative_dynamic_auc(y_train, new_y_test, risk_scores, times) \n",
    "                    preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])\n",
    "                    IBS = integrated_brier_score(y_train, new_y_test, preds, times)    \n",
    "\n",
    "                    df.loc[i, f'{tx}_Tx_name'] = tx_name\n",
    "                    df.loc[i, f'{tx}_C_index'] = c_index\n",
    "                    df.loc[i, f'{tx}_Mean_auc'] = mean_auc\n",
    "                    df.loc[i, f'{tx}_IBS'] = IBS\n",
    "\n",
    "                except:\n",
    "                    ValueError\n",
    "                    df.loc[i, f'{tx}_Tx_name'] = tx_name\n",
    "                    df.loc[i, f'{tx}_C_index'] = 'Value error'\n",
    "                    df.loc[i, f'{tx}_Mean_auc'] = 'Value error'\n",
    "                    df.loc[i, f'{tx}_IBS'] = 'Value error'                \n",
    "            \n",
    "    df_all = pd.concat([df_all, df])"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
