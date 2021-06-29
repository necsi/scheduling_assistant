# scheduling_assistant

Create a schedule pairing up teams across timezones.

Created for the Global Summit 2021



## Usage
  1. Update teams.csv with real teams, their domains and timezones. 
  2. Update init_schedule.csv with events for all teams.
  3. Update time_of_day_constraints.csv if desired for availability based on time of day (rating 0-3 where 0 is unavailable, 3 is best time)
  4. Run: python scheduling_assistant.py 
  5. Outputs: team_schedule.csv (meetings by team), full_schedule.csv (all events by time)

#### Notes
There is a false positive error "SettingWithCopyWarning", please ignore.
