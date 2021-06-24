# scheduling_assistant

Create a schedule pairing up teams across timezones.

Created for the Global Summit 2021



## Usage
  1. Update teams.csv with real teams, their domains and timezones. 
  
  2. Update time_of_day_constraints.csv if desired for availability based on time of day (rating 0-3 where 0 is unavailable, 3 is best time)
  
  3. Run python scheduling_assistant.py 
  
  4. Output: schedule.csv
  
 There is a small chance this will produce a non-optimal schedule, for instance a team being selected to be unpaired more than once. Please check the output and re-run for better results.

