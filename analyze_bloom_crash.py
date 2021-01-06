# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:15:03 2020

@author: ricro
"""

"""
Analyzing Bloomington, IN crash data from 2003 - 2015

"""

"""

Looking at crash data for Monroe County (Bloomington, IN) from 2003 to 2015

Variables: (in order in which they appear in columns)
(0) Master Record Number
(1) Year
(2) Month
(3) Day
(4) Weekend?
(5) Hour (in military time)
(6) Collision Type
(7) Injury Type
(8) Primary Factor (suspected reason for crash)
(9) Reported Location (Street1 & Street2)
(10) Latitude
(11) Longitude

12 columns in total

Some questions to ask:

Look at distribution of values for applicable variables
Which months have the most amount of crashes?
Weekend or weekday more likely to have a crash? 
How far from some central location (distribution)? 
Say, Indiana Memorial Union, or Sample Gates? 
Any correlated variables?
Most often reason given for crash? 
Distribution of how many cars involved in accident? 
Are the injury types distributed uniformly across months? across years? 



"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

filename = "crashData.txt"

crash_data = pd.read_csv(filename, encoding = 'latin1')

crash_data.head()

# Let's look at a count of:
# (1) InjuryType
# (2) PrimaryFactor

SampleGates = np.array([39.166550, -86.526380]) # Latitude and longitude, 
# resp., of a reference point in Bloomington, IN
# (Sample Gates: main entrance to Indiana University Bloomington, 
# at the end of Kirkwood Avenue.) 

# Let's also look at the distribution of distance to the Sample Gates;
# Since Bloomington is a small area relative to the whole earth, we can approximate distance fairly accurately using the 
# basic Euclidean distance

Lat = crash_data['Latitude']
Lon = crash_data['Longitude']

Distance = np.sqrt( (Lat - SampleGates[0])**2 + (Lon - SampleGates[1])**2)

# Now let's plot the distribution of distances:
plt.hist(Distance[Distance <= 25],bins=100)
plt.title('Distribution of Distances of Crashes from Sample Gates, Bloomington, IN')
# There are a few outliers close to a distance of 100 away from Sample Gates; what does the distribution look like if we exclude these?

PropNearGates = sum(Distance <= 1)/len(Distance)
# Approximately 91% of the crashes occur within 1 unit of distance from Sample Gates;
# That is, nearly 91% of the crashes in the city of Bloomington happen around the university

print(PropNearGates)

# Let's plot a map of where these crashes occur:
fig, ax = plt.subplots(1,1)
ax.plot(Lat, Lon,'bd',SampleGates[0], SampleGates[1],'r*')
ax.set_title("Spread of Crashes relative to Sample Gates")
# Focusing on crashes near the university:
ax.set_xlim([35, 45])
ax.set_ylim([-90, -80])

## Looking at the primary factors for the crashes:
Primary = crash_data['PrimaryFactor']
print("There were {x} unique reasons for the {y} crashes that occurred between 2003 and 2015 in Bloomington, IN".format(x=len(Primary.unique()), 
                                                                                                                        y=len(Primary)))
Primary.count() # 53,943 crashes between 2004 and 2015 in B-Town, IN
# Note that "<undefined>" values indicate that the reason was not recorded, 
# or that someone ran from the scene of the accident
PrimaryCounts = Primary.value_counts()

N_crashes = len(Primary)

fig = plt.figure(figsize=(8,12), dpi = 100)
plt.barh(PrimaryCounts.index[::-1], PrimaryCounts.values[::-1]/N_crashes)
plt.xticks(rotation=90)
plt.title('Proportion of each Reason for Crashes')


# So, we have so far that most crashes happen quite close to the university,
# and the most common reasons for the crash include a failure 
# to yield the right of way and following too closely. 


### ANALYSES ACROSS DIFFERENT MONTHS/DAYS

# Because Bloomington is a college town, much of the traffic would be expected
# to revolve around the cycle of the semesters for the university.
# First, how have the amount of crashes in town changed over the years?

CrashesPerYear = crash_data['Year']
CrashesPerYear = CrashesPerYear.value_counts()
CrashesPerYear = CrashesPerYear.sort_index()

fig, ax = plt.subplots(1,1)
ax.plot(CrashesPerYear.index.values, CrashesPerYear.values, 'bo-')
ax.set_title('Recorded Crashes Per Year')

# Doesn't seem to have a clear pattern, other than the largest value being 
# the first recorded year in this dataset. Somewhat cyclical pattern for 
# some reason. 

# What about looking at the average per month? 

# Here, we're organizing by Year and extracting only the years and the months
MonthCrashes = crash_data[['Year','Month']]
MonthCrashes = MonthCrashes.groupby(['Year','Month']).size().reset_index()
MonthCrashes = MonthCrashes.rename(columns = {0: 'counts'})
fig, ax = plt.subplots(1,1)

Mapping = {1:"Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7:"Jul",
          8: "Aug", 9: "Sep", 10:"Oct", 11:"Nov", 12: "Dec"}

years = MonthCrashes['Year'].unique()
for year in years:
    Idx = MonthCrashes['Year'] == year
    data = MonthCrashes[Idx][['Month','counts']]
    ax.plot(data['Month'].map(Mapping), data['counts'],'-')
    
ax.set_title('Crashes Per Month for Each Year')
ax.set_xlabel('Year')
ax.set_ylabel('Crashes')
ax.set_ylim([0, 500])

# Get Mean and SD for each month, across the years
MonthAgg = MonthCrashes[['Month','counts']].groupby('Month').agg(['mean', 'std'])
fig, ax = plt.subplots(1,1)
ax.errorbar(MonthAgg.index.map(Mapping).values, 
            MonthAgg['counts']['mean'].values, 
            yerr = MonthAgg['counts']['std'].values)
ax.set_ylim([0, 500])
ax.set_title('Mean (SD) of Crashes per Month, Across Years')
ax.set_xlabel('Month')
ax.set_ylabel('Mean Crashes')

# For some reason, the number of crashes appear to be maximal in October
# Indeed, in the Fall semesters, starting in August, the amount of crashes 
# seem to steadily increase up to October, and then decline as the semester
# ends

# We can check if there are trends in the data by trying an ANOVA analysis
# on the two semesters: Spring (Jan - May) and Fall (August - December)

# Separating data into Fall and Spring months:
fall = np.array([1,2,3,4,5])
spring = np.array([8,9,10,11,12])
MonthCounts = MonthCrashes[['Month','counts']]

Fall_DF = pd.DataFrame()
Spring_DF = pd.DataFrame()

for f in fall:
    Idx = MonthCounts['Month'] == f
    Fall_DF = Fall_DF.append(MonthCounts[Idx])
    
for s in spring:
    Idx = MonthCounts['Month'] == s
    Spring_DF = Spring_DF.append(MonthCounts[Idx])


Fall_DF.reset_index()
Spring_DF.reset_index()

N = len(fall)

fall_data = {}
spring_data = {}

for n in range(N):
    IdxF = Fall_DF['Month'] == fall[n]
    IdxS = Spring_DF['Month'] == spring[n]
    fall_data[n] = Fall_DF[IdxF.values]['counts'].values
    spring_data[n] = Spring_DF[IdxS.values]['counts'].values

fall_aov = stats.f_oneway(fall_data[0], fall_data[1], fall_data[2], 
                          fall_data[3], fall_data[4])

spring_aov = stats.f_oneway(spring_data[0], spring_data[1], spring_data[2], 
                          spring_data[3], spring_data[4])

print("Fall:")
print(fall_aov)

print("")
print("Spring:")
print(spring_aov)


# October is clearly the mot dangerous month, consistenly across all these sampled
# years; but is the most likely reason for the crashes the same in October
# as it is when considering all months together?

# That is, if we let X = most often cited reason for crash, then do we have:
# P(X) = P(X | October)? 

oct_data = crash_data[['Month', 'PrimaryFactor']]
october = crash_data['Month'] == 10
oct_data = oct_data[october]

oct_data = oct_data['PrimaryFactor']

October_Reasons = oct_data.value_counts()

print('')
print(October_Reasons.head())

# As can be seen from the value counts, it seems like the october-only 
# data have a nearly identical distribution from the overall crash data, in 
# terms of the reasons for the crashes. 

Compare_Probs = pd.merge(PrimaryCounts/sum(PrimaryCounts),
                         October_Reasons/sum(October_Reasons), 
                         how = 'outer',
                         left_index = True,
                         right_index = True,
                         suffixes = ("_Year","_Oct"))

Compare_Probs = Compare_Probs.sort_values(by = 'PrimaryFactor_Year',
                                          ascending = False)

print(Compare_Probs.head())

# So it appears that in fact we have that P(X) = P(X | October). That is,
# month of October is probabilistically more or less identical to the pattern
# for the entire year. Nonetheless, the raw counts of accidents consistently
# show October to be the most dangerous month of the year when it comes to 
# car accidents. In fact, we consistently see a steady increase in accidents from
# August into October, followed by a drop off in November and December. 


# Overall, we can conclude here:
    # The most dangerous month in Bloomington for driving is in October,
    # with the most likely reason for a crash being someone failing to yield
    # the right of way, and indeed these accidents are most likely to occur 
    # very near the main entrance of the university (Sample Gates on Kirkwood
    # and Indiana). 



