# scheduling_assistant

Created for the Global Summit 2021

* Create a schedule pairing up teams across timezones.
* Find speaker availability based on time zone and meeting time.





## Usage
  For scheduling team/team pair meetings:
  1. Update teams_example.csv with real teams, their domains and timezones. 
  2. Update init_schedule_example.csv with events for all teams.
  3. Update speakers_example.csv with real speakers
  4. Update time_of_day_constraints.csv for availability based on time of day (rating 0-3 where 0 is unavailable, 3 is best time)
  5. Run: python scheduling_assistant.py 
  * Optional parameters
    * --teams: Input teams spreadsheet
    * --tod: Input time of day weights spreadsheet
    * --main: Input main events schedule spreadsheet
    * --speakers: Input speakers spreadsheet
    * --trials: How many combinations to try when matching. Default 10,000
    
  7. Outputs: team_schedule.csv (meetings by team), full_schedule.csv (all events in calendar format)
  
  For finding speaker availability:
  1. Update speakers.csv with speaker name, topic, and timezone
  2. Make sure init_schedule.csv has Skills blocks (Skills 1, Skills 2, Skills 3)
  3. Run: python scheduling_assistant.py
  4. Outputs: speaker_schedule.csv 

#### Notes
There is a false positive error "SettingWithCopyWarning", please ignore.
