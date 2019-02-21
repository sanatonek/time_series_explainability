# Generates the following data file from MIMIC:
# adult_icu.gz: data from adult ICUs

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import ks_2samp
import os 
import random
import argparse


def replace(group):
  """ Replace missing values in measurements using mean imputation
  takes in a pandas group, and replaces the null value with the mean of the none null
  values of the same group 
  """
  mask = group.isnull()
  group[mask] = group[~mask].mean()
  return group


def main(sqluser, sqlpass):
  random.seed(22891)
  # Ouput directory to generate the files
  mimicdir = os.path.expanduser("./")

  # create a database connection and connect to local postgres version of mimic
  dbname = 'mimic'
  schema_name = 'mimiciii'
  con = psycopg2.connect(dbname=dbname, user=sqluser, host='127.0.0.1', password=sqlpass)
  cur = con.cursor()
  cur.execute('SET search_path to ' + schema_name)



  #========get the icu details 

  # this query extracts the following:
  #   Unique ids for the admission, patient and icu stay 
  #   Patient gender 
  #   admission & discharge times 
  #   length of stay 
  #   age 
  #   ethnicity 
  #   admission type 
  #   in hospital death?
  #   in icu death?
  #   one year from admission death?
  #   first hospital stay 
  #   icu intime, icu outime 
  #   los in icu 
  #   first icu stay?

  denquery = \
  """
  -- This query extracts useful demographic/administrative information for patient ICU stays
  --DROP MATERIALIZED VIEW IF EXISTS icustay_detail CASCADE;
  --CREATE MATERIALIZED VIEW icustay_detail as

  --ie is the icustays table 
  --adm is the admissions table 
  SELECT ie.subject_id, ie.hadm_id, ie.icustay_id
  , pat.gender
  , adm.admittime, adm.dischtime, adm.diagnosis
  , ROUND( (CAST(adm.dischtime AS DATE) - CAST(adm.admittime AS DATE)) , 4) AS los_hospital
  , ROUND( (CAST(adm.admittime AS DATE) - CAST(pat.dob AS DATE))  / 365, 4) AS age
  , adm.ethnicity, adm.ADMISSION_TYPE
  --, adm.hospital_expire_flag
  , CASE when adm.deathtime between ie.intime and ie.outtime THEN 1 ELSE 0 END AS mort_icu
  , DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS hospstay_seq
  , CASE
      WHEN DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) = 1 THEN 1
      ELSE 0 END AS first_hosp_stay
  -- icu level factors
  , ie.intime, ie.outtime
  , ie.FIRST_CAREUNIT
  , ROUND( (CAST(ie.outtime AS DATE) - CAST(ie.intime AS DATE)) , 4) AS los_icu
  , DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS icustay_seq

  -- first ICU stay *for the current hospitalization*
  , CASE
      WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN 1
      ELSE 0 END AS first_icu_stay

  FROM icustays ie
  INNER JOIN admissions adm
      ON ie.hadm_id = adm.hadm_id
  INNER JOIN patients pat
      ON ie.subject_id = pat.subject_id
  WHERE adm.has_chartevents_data = 1
  ORDER BY ie.subject_id, adm.admittime, ie.intime;

  """

  den = pd.read_sql_query(denquery,con)

  #----drop patients with less than 48 hour 
  den['los_icu_hr'] = (den.outtime - den.intime).astype('timedelta64[h]')
  den = den[(den.los_icu_hr >= 48)]
  den = den[(den.age<300)]
  den.drop('los_icu_hr', 1, inplace = True)
  # den.isnull().sum()

  #----clean up

  # micu --> medical 
  # csru --> cardiac surgery recovery unit 
  # sicu --> surgical icu 
  # tsicu --> Trauma Surgical Intensive Care Unit
  # NICU --> Neonatal 

  den['adult_icu'] = np.where(den['first_careunit'].isin(['PICU', 'NICU']), 0, 1)
  den['gender'] = np.where(den['gender']=="M", 1, 0)
  den.ethnicity = den.ethnicity.str.lower()
  den.ethnicity.loc[(den.ethnicity.str.contains('^white'))] = 'white'
  den.ethnicity.loc[(den.ethnicity.str.contains('^black'))] = 'black'
  den.ethnicity.loc[(den.ethnicity.str.contains('^hisp')) | (den.ethnicity.str.contains('^latin'))] = 'hispanic'
  den.ethnicity.loc[(den.ethnicity.str.contains('^asia'))] = 'asian'
  den.ethnicity.loc[~(den.ethnicity.str.contains('|'.join(['white', 'black', 'hispanic', 'asian'])))] = 'other'
  den = pd.concat([den, pd.get_dummies(den['ethnicity'], prefix='eth')], 1)
  den = pd.concat([den, pd.get_dummies(den['admission_type'], prefix='admType')], 1)

  den.drop(['diagnosis', 'hospstay_seq', 'los_icu','icustay_seq', 'admittime', 'dischtime','los_hospital', 'intime', 'outtime', 'ethnicity', 'admission_type', 'first_careunit'], 1, inplace =True) 

  #========= 48 hour vitals query 
  # these are the normal ranges. useful to clean up the data

  vitquery = \
  """
  -- This query pivots the vital signs for the first 48 hours of a patient's stay
  -- Vital signs include heart rate, blood pressure, respiration rate, and temperature
  -- DROP MATERIALIZED VIEW IF EXISTS vitalsfirstday CASCADE;
  -- create materialized view vitalsfirstday as
  SELECT pvt.subject_id, pvt.hadm_id, pvt.icustay_id

  -- Easier names
  , min(case when VitalID = 1 then valuenum else null end) as HeartRate_Min
  , max(case when VitalID = 1 then valuenum else null end) as HeartRate_Max
  , avg(case when VitalID = 1 then valuenum else null end) as HeartRate_Mean
  , min(case when VitalID = 2 then valuenum else null end) as SysBP_Min
  , max(case when VitalID = 2 then valuenum else null end) as SysBP_Max
  , avg(case when VitalID = 2 then valuenum else null end) as SysBP_Mean
  , min(case when VitalID = 3 then valuenum else null end) as DiasBP_Min
  , max(case when VitalID = 3 then valuenum else null end) as DiasBP_Max
  , avg(case when VitalID = 3 then valuenum else null end) as DiasBP_Mean
  , min(case when VitalID = 4 then valuenum else null end) as MeanBP_Min
  , max(case when VitalID = 4 then valuenum else null end) as MeanBP_Max
  , avg(case when VitalID = 4 then valuenum else null end) as MeanBP_Mean
  , min(case when VitalID = 5 then valuenum else null end) as RespRate_Min
  , max(case when VitalID = 5 then valuenum else null end) as RespRate_Max
  , avg(case when VitalID = 5 then valuenum else null end) as RespRate_Mean
  , min(case when VitalID = 6 then valuenum else null end) as TempC_Min
  , max(case when VitalID = 6 then valuenum else null end) as TempC_Max
  , avg(case when VitalID = 6 then valuenum else null end) as TempC_Mean
  , min(case when VitalID = 7 then valuenum else null end) as SpO2_Min
  , max(case when VitalID = 7 then valuenum else null end) as SpO2_Max
  , avg(case when VitalID = 7 then valuenum else null end) as SpO2_Mean
  , min(case when VitalID = 8 then valuenum else null end) as Glucose_Min
  , max(case when VitalID = 8 then valuenum else null end) as Glucose_Max
  , avg(case when VitalID = 8 then valuenum else null end) as Glucose_Mean

  FROM  (
    select ie.subject_id, ie.hadm_id, ie.icustay_id, ce.charttime, ce.valuenum
    , case
      when itemid in (211,220045) and valuenum > 0 and valuenum < 300 then 1 -- HeartRate
      when itemid in (51,442,455,6701,220179,220050) and valuenum > 0 and valuenum < 400 then 2 -- SysBP
      when itemid in (8368,8440,8441,8555,220180,220051) and valuenum > 0 and valuenum < 300 then 3 -- DiasBP
      when itemid in (456,52,6702,443,220052,220181,225312) and valuenum > 0 and valuenum < 300 then 4 -- MeanBP
      when itemid in (615,618,220210,224690) and valuenum > 0 and valuenum < 70 then 5 -- RespRate
      when itemid in (223761,678) and valuenum > 70 and valuenum < 120  then 6 -- TempF, converted to degC in valuenum call
      when itemid in (223762,676) and valuenum > 10 and valuenum < 50  then 6 -- TempC
      when itemid in (646,220277) and valuenum > 0 and valuenum <= 100 then 7 -- SpO2
      when itemid in (807,811,1529,3745,3744,225664,220621,226537) and valuenum > 0 then 8 -- Glucose

      else null end as VitalID
        -- convert F to C
    , case when itemid in (223761,678) then (valuenum-32)/1.8 else valuenum end as valuenum

    from icustays ie
    left join chartevents ce
    on ie.subject_id = ce.subject_id and ie.hadm_id = ce.hadm_id and ie.icustay_id = ce.icustay_id
    and ce.charttime between ie.intime and ie.intime + interval '48' hour
    -- exclude rows marked as error
    and ce.error IS DISTINCT FROM 1
    where ce.itemid in
    (
    -- HEART RATE
    211, --"Heart Rate"
    220045, --"Heart Rate"

    -- Systolic/diastolic

    51, --	Arterial BP [Systolic]
    442, --	Manual BP [Systolic]
    455, --	NBP [Systolic]
    6701, --	Arterial BP #2 [Systolic]
    220179, --	Non Invasive Blood Pressure systolic
    220050, --	Arterial Blood Pressure systolic

    8368, --	Arterial BP [Diastolic]
    8440, --	Manual BP [Diastolic]
    8441, --	NBP [Diastolic]
    8555, --	Arterial BP #2 [Diastolic]
    220180, --	Non Invasive Blood Pressure diastolic
    220051, --	Arterial Blood Pressure diastolic


    -- MEAN ARTERIAL PRESSURE
    456, --"NBP Mean"
    52, --"Arterial BP Mean"
    6702, --	Arterial BP Mean #2
    443, --	Manual BP Mean(calc)
    220052, --"Arterial Blood Pressure mean"
    220181, --"Non Invasive Blood Pressure mean"
    225312, --"ART BP mean"

    -- RESPIRATORY RATE
    618,--	Respiratory Rate
    615,--	Resp Rate (Total)
    220210,--	Respiratory Rate
    224690, --	Respiratory Rate (Total)


    -- SPO2, peripheral
    646, 220277,

    -- GLUCOSE, both lab and fingerstick
    807,--	Fingerstick Glucose
    811,--	Glucose (70-105)
    1529,--	Glucose
    3745,--	BloodGlucose
    3744,--	Blood Glucose
    225664,--	Glucose finger stick
    220621,--	Glucose (serum)
    226537,--	Glucose (whole blood)

    -- TEMPERATURE
    223762, -- "Temperature Celsius"
    676,	-- "Temperature C"
    223761, -- "Temperature Fahrenheit"
    678 --	"Temperature F"

    )
  ) pvt
  group by pvt.subject_id, pvt.hadm_id, pvt.icustay_id
  order by pvt.subject_id, pvt.hadm_id, pvt.icustay_id;
  """

  vit48 = pd.read_sql_query(vitquery,con)
  vit48.isnull().sum()


  #===============48 hour labs query 
  # This query extracts the lab events in the first 48 hours 
  labquery = \
  """
  WITH pvt AS (
    --- ie is the icu stay 
    --- ad is the admissions table 
    --- le is the lab events table 
    SELECT ie.subject_id, ie.hadm_id, ie.icustay_id, le.charttime
    -- here we assign labels to ITEMIDs
    -- this also fuses together multiple ITEMIDs containing the same data
    , CASE
          when le.itemid = 50868 then 'ANION GAP'
          when le.itemid = 50862 then 'ALBUMIN'
          when le.itemid = 50882 then 'BICARBONATE'
          when le.itemid = 50885 then 'BILIRUBIN'
          when le.itemid = 50912 then 'CREATININE'
          when le.itemid = 50806 then 'CHLORIDE'
          when le.itemid = 50902 then 'CHLORIDE'
          when le.itemid = 50809 then 'GLUCOSE'
          when le.itemid = 50931 then 'GLUCOSE'
          when le.itemid = 50810 then 'HEMATOCRIT'
          when le.itemid = 51221 then 'HEMATOCRIT'
          when le.itemid = 50811 then 'HEMOGLOBIN'
          when le.itemid = 51222 then 'HEMOGLOBIN'
          when le.itemid = 50813 then 'LACTATE'
          when le.itemid = 50960 then 'MAGNESIUM'
          when le.itemid = 50970 then 'PHOSPHATE'
          when le.itemid = 51265 then 'PLATELET'
          when le.itemid = 50822 then 'POTASSIUM'
          when le.itemid = 50971 then 'POTASSIUM'
          when le.itemid = 51275 then 'PTT'
          when le.itemid = 51237 then 'INR'
          when le.itemid = 51274 then 'PT'
          when le.itemid = 50824 then 'SODIUM'
          when le.itemid = 50983 then 'SODIUM'
          when le.itemid = 51006 then 'BUN'
          when le.itemid = 51300 then 'WBC'
          when le.itemid = 51301 then 'WBC'
        ELSE null
        END AS label
    , -- add in some sanity checks on the values
      -- the where clause below requires all valuenum to be > 0, 
      -- so these are only upper limit checks
      CASE
        when le.itemid = 50862 and le.valuenum >    10 then null -- g/dL 'ALBUMIN'
        when le.itemid = 50868 and le.valuenum > 10000 then null -- mEq/L 'ANION GAP'
        when le.itemid = 50882 and le.valuenum > 10000 then null -- mEq/L 'BICARBONATE'
        when le.itemid = 50885 and le.valuenum >   150 then null -- mg/dL 'BILIRUBIN'
        when le.itemid = 50806 and le.valuenum > 10000 then null -- mEq/L 'CHLORIDE'
        when le.itemid = 50902 and le.valuenum > 10000 then null -- mEq/L 'CHLORIDE'
        when le.itemid = 50912 and le.valuenum >   150 then null -- mg/dL 'CREATININE'
        when le.itemid = 50809 and le.valuenum > 10000 then null -- mg/dL 'GLUCOSE'
        when le.itemid = 50931 and le.valuenum > 10000 then null -- mg/dL 'GLUCOSE'
        when le.itemid = 50810 and le.valuenum >   100 then null -- % 'HEMATOCRIT'
        when le.itemid = 51221 and le.valuenum >   100 then null -- % 'HEMATOCRIT'
        when le.itemid = 50811 and le.valuenum >    50 then null -- g/dL 'HEMOGLOBIN'
        when le.itemid = 51222 and le.valuenum >    50 then null -- g/dL 'HEMOGLOBIN'
        when le.itemid = 50813 and le.valuenum >    50 then null -- mmol/L 'LACTATE'
        when le.itemid = 50960 and le.valuenum >    60 then null -- mmol/L 'MAGNESIUM'
        when le.itemid = 50970 and le.valuenum >    60 then null -- mg/dL 'PHOSPHATE'
        when le.itemid = 51265 and le.valuenum > 10000 then null -- K/uL 'PLATELET'
        when le.itemid = 50822 and le.valuenum >    30 then null -- mEq/L 'POTASSIUM'
        when le.itemid = 50971 and le.valuenum >    30 then null -- mEq/L 'POTASSIUM'
        when le.itemid = 51275 and le.valuenum >   150 then null -- sec 'PTT'
        when le.itemid = 51237 and le.valuenum >    50 then null -- 'INR'
        when le.itemid = 51274 and le.valuenum >   150 then null -- sec 'PT'
        when le.itemid = 50824 and le.valuenum >   200 then null -- mEq/L == mmol/L 'SODIUM'
        when le.itemid = 50983 and le.valuenum >   200 then null -- mEq/L == mmol/L 'SODIUM'
        when le.itemid = 51006 and le.valuenum >   300 then null -- 'BUN'
        when le.itemid = 51300 and le.valuenum >  1000 then null -- 'WBC'
        when le.itemid = 51301 and le.valuenum >  1000 then null -- 'WBC'
      ELSE le.valuenum
      END AS valuenum
    FROM icustays ie

    LEFT JOIN labevents le
      ON le.subject_id = ie.subject_id 
      AND le.hadm_id = ie.hadm_id
      -- TODO: they are using lab times 6 hours before the start of the 
      -- ICU stay. 
      AND le.charttime between (ie.intime - interval '6' hour) 
      AND (ie.intime + interval '48' hour)
      AND le.itemid IN
      (
        -- comment is: LABEL | CATEGORY | FLUID | NUMBER OF ROWS IN LABEVENTS
        50868, -- ANION GAP | CHEMISTRY | BLOOD | 769895
        50862, -- ALBUMIN | CHEMISTRY | BLOOD | 146697
        50882, -- BICARBONATE | CHEMISTRY | BLOOD | 780733
        50885, -- BILIRUBIN, TOTAL | CHEMISTRY | BLOOD | 238277
        50912, -- CREATININE | CHEMISTRY | BLOOD | 797476
        50902, -- CHLORIDE | CHEMISTRY | BLOOD | 795568
        50806, -- CHLORIDE, WHOLE BLOOD | BLOOD GAS | BLOOD | 48187
        50931, -- GLUCOSE | CHEMISTRY | BLOOD | 748981
        50809, -- GLUCOSE | BLOOD GAS | BLOOD | 196734
        51221, -- HEMATOCRIT | HEMATOLOGY | BLOOD | 881846
        50810, -- HEMATOCRIT, CALCULATED | BLOOD GAS | BLOOD | 89715
        51222, -- HEMOGLOBIN | HEMATOLOGY | BLOOD | 752523
        50811, -- HEMOGLOBIN | BLOOD GAS | BLOOD | 89712
        50813, -- LACTATE | BLOOD GAS | BLOOD | 187124
        50960, -- MAGNESIUM | CHEMISTRY | BLOOD | 664191
        50970, -- PHOSPHATE | CHEMISTRY | BLOOD | 590524
        51265, -- PLATELET COUNT | HEMATOLOGY | BLOOD | 778444
        50971, -- POTASSIUM | CHEMISTRY | BLOOD | 845825
        50822, -- POTASSIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 192946
        51275, -- PTT | HEMATOLOGY | BLOOD | 474937
        51237, -- INR(PT) | HEMATOLOGY | BLOOD | 471183
        51274, -- PT | HEMATOLOGY | BLOOD | 469090
        50983, -- SODIUM | CHEMISTRY | BLOOD | 808489
        50824, -- SODIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 71503
        51006, -- UREA NITROGEN | CHEMISTRY | BLOOD | 791925
        51301, -- WHITE BLOOD CELLS | HEMATOLOGY | BLOOD | 753301
        51300  -- WBC COUNT | HEMATOLOGY | BLOOD | 2371
      )
      AND le.valuenum IS NOT null 
      AND le.valuenum > 0 -- lab values cannot be 0 and cannot be negative
      
      LEFT JOIN admissions ad
      ON ie.subject_id = ad.subject_id
      AND ie.hadm_id = ad.hadm_id

      
  ),
  ranked AS (
  SELECT pvt.*, DENSE_RANK() OVER (PARTITION BY 
      pvt.subject_id, pvt.hadm_id,pvt.icustay_id,pvt.label ORDER BY pvt.charttime) as drank
  FROM pvt
  )
  SELECT r.subject_id, r.hadm_id, r.icustay_id
    , max(case when label = 'ANION GAP' then valuenum else null end) as ANIONGAP
    , max(case when label = 'ALBUMIN' then valuenum else null end) as ALBUMIN
    , max(case when label = 'BICARBONATE' then valuenum else null end) as BICARBONATE
    , max(case when label = 'BILIRUBIN' then valuenum else null end) as BILIRUBIN
    , max(case when label = 'CREATININE' then valuenum else null end) as CREATININE
    , max(case when label = 'CHLORIDE' then valuenum else null end) as CHLORIDE
    , max(case when label = 'GLUCOSE' then valuenum else null end) as GLUCOSE
    , max(case when label = 'HEMATOCRIT' then valuenum else null end) as HEMATOCRIT
    , max(case when label = 'HEMOGLOBIN' then valuenum else null end) as HEMOGLOBIN
    , max(case when label = 'LACTATE' then valuenum else null end) as LACTATE
    , max(case when label = 'MAGNESIUM' then valuenum else null end) as MAGNESIUM
    , max(case when label = 'PHOSPHATE' then valuenum else null end) as PHOSPHATE
    , max(case when label = 'PLATELET' then valuenum else null end) as PLATELET
    , max(case when label = 'POTASSIUM' then valuenum else null end) as POTASSIUM
    , max(case when label = 'PTT' then valuenum else null end) as PTT
    , max(case when label = 'INR' then valuenum else null end) as INR
    , max(case when label = 'PT' then valuenum else null end) as PT
    , max(case when label = 'SODIUM' then valuenum else null end) as SODIUM
    , max(case when label = 'BUN' then valuenum else null end) as BUN
    , max(case when label = 'WBC' then valuenum else null end) as WBC

  FROM ranked r
  WHERE r.drank = 1
  GROUP BY r.subject_id, r.hadm_id, r.icustay_id,  r.drank
  ORDER BY r.subject_id, r.hadm_id, r.icustay_id,  r.drank;
  """

  lab48 = pd.read_sql_query(labquery,con)


  #=====combine all variables 
  mort_ds = den.merge(vit48,how = 'left',    on = ['subject_id', 'hadm_id', 'icustay_id'])
  mort_ds = mort_ds.merge(lab48,how = 'left',    on = ['subject_id', 'hadm_id', 'icustay_id'])


  # create means by age group and gender 
  mort_ds['age_group'] = pd.cut(mort_ds['age'], [-1,5,10,15,20, 25, 40,60, 80, 200], 
    labels = ['l5','5_10', '10_15', '15_20', '20_25', '25_40', '40_60',  '60_80', '80p'])


  mort_ds = mort_ds.groupby(['age_group', 'gender'])
  mort_ds = mort_ds.transform(replace)

  # one missing variable 
  adult_icu = mort_ds[(mort_ds.adult_icu==1)].dropna()
  # create training and testing labels 
  msk = np.random.rand(len(adult_icu)) < 0.7
  adult_icu['train'] = np.where(msk, 1, 0) 
  adult_icu.to_csv(os.path.join(mimicdir, 'adult_icu.gz'), compression='gzip',  index = False)





if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Query ICU mortality data from mimic database')
  parser.add_argument('--sqluser',type=str, default='mimicuser' ,help='postgres user to access mimic database')
  parser.add_argument('--sqlpass',type=str , default='PASSWORD', help='postgres user password to access mimic database')
  args = parser.parse_args()
  main(args.sqluser, args.sqlpass)
