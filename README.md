# scheduling_assistant

Create a schedule pairing up teams across timezones.

Created for the Global Summit 2021

There is a small chance this will produce a non-optimal schedule, for instance a team being selected to be unpaired more than once. Please check it and re-run for better results.

Usage: 
  Update teams.csv with real teams, their domains and timezones
  Update time_of_day_constraints.csv if desired for availability based on time of day (rating 0-3 where 0 is unavailable, 3 is best time)
  
  python scheduling_assistant.py 
  
  Output: schedule.csv

