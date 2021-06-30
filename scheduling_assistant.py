from datetime import time
from os import pardir
from typing import ByteString
from numpy.core.numerictypes import maximum_sctype
from ortools.sat.python.cp_model import Domain
import pandas as pd
import numpy as np
import csv
import argparse


def printMatches(all_matches,  team_df):

    print("Team\t\tDomain Mtg 1\tDomain Mtg 2\tMeeting 3\tMeeting 4")
    n_teams = team_df.shape[0]
    n_events = all_matches.shape[0]
    for t1 in range(n_teams):
        print("%s (%s)" % (team_df['Team'][t1], team_df['Domain'][t1][:3]), ":", end="")
        for e in range(n_events):
            t2 = all_matches[e][t1]
            if t2 >= 0:
                print("\t%s (%s)" % (team_df['Team'][t2], team_df['Domain'][t2][:3]), end="\t")
            else: 
                print("\tNone\t", end="")
        print()

def exportMatches(all_matches, suggested_times, team_df, tod_constraints, main_schedule_df, filename='matches.csv'):
    # Exports a per-team schedule with matching teams and times

    headers = ['Team']
    for m in range(1, all_matches.shape[0] + 1):
        headers.append('Meeting %i' % (m))
        headers.append('Time %i (UTC)' % (m))
        headers.append('Time %i (Local)' % (m))
    n_teams = team_df.shape[0]
    all_matches = all_matches.T


    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for t1 in range(n_teams):
            row = [team_df['Team'][t1]]
            for m in range(all_matches[t1].shape[0]):
                t2 = int(all_matches[t1, m])
                if t2 >= 0:
                    row.append(team_df['Team'][t2])
                    suggested_timeslot = suggested_times[t1, t2]

                    suggested_time_utc = main_schedule_df['Datetime'][int(suggested_timeslot)]
                    row.append(suggested_time_utc.strftime('%m/%d/%Y %H:%M'))
                    timezone = team_df['Timezone'][t1]
                    local_datetime = suggested_time_utc + pd.Timedelta(hours=int(timezone))
                    local_datetime_str = local_datetime.strftime('%m/%d/%Y %H:%M')
                    row.append(local_datetime_str)
                else:
                    row.append('None')
                    row.append('')
                    row.append('')
            writer.writerow(row)
    print('wrote file', filename, "\n")


def exportFullSchedule(main_schedule_df, teams_df, all_matches, suggested_times, filename='full_schedule.csv'):
    # Exports a calendar-style schedule of all events


    full_df = main_schedule_df.copy()
    for i, team in enumerate(teams_df['Team']):
        full_df[team] = main_schedule_df['Events']
        for m in range(len(all_matches)):
            meeting_partner = all_matches[m][i]
            if(meeting_partner >= 0):
                meeting_timeslot = suggested_times[i][meeting_partner]
                full_df[team].iloc[meeting_timeslot] = teams_df['Team'].iloc[meeting_partner]

    full_df.drop(columns=['Timeslot', 'Events'], inplace=True)
    full_df.set_index('Datetime', inplace=True, drop=True)


    full_df = full_df.T

    full_df.set_index(np.arange(full_df.shape[0] ), inplace=True)
    full_df['Timezone'] = teams_df['Timezone']
    full_df['Team'] = teams_df['Team']
  

    cols = full_df.columns.tolist()
    cols = cols[:-2]
    cols = ['Team', 'Timezone'] + cols
    full_df = full_df[cols]
 

    
    full_df.to_csv(filename, index=False)
    print("\nwrote file", filename, "\n")



def initAvailability(teams_df, main_schedule_df, tod_constraints, initial_hour, n_timeslots, slots_per_hour):
 
    # Returns time weights for each team, based on time of day constraints and main schedule constraints
    
    slots_per_hour = 2 # meetings available on the half hour
    main_availability = main_schedule_df['Events'].isna().astype(int).to_numpy() # Not available when an all teams event is scheduled

    availability = []
    timezones = teams_df['Timezone'].to_list()
    for tz in timezones:
        team_availability = np.roll(tod_constraints, -tz*slots_per_hour)
        team_availability = team_availability[initial_hour*slots_per_hour:]
        team_availability = team_availability[:n_timeslots]
        team_availability *= main_availability

        availability.append(team_availability)

    return np.array(availability)



def doodle(availability):
    # Doodle poll-like function to find best times given availabilities defined as a weight for each hour, higher weights better
    # Returns array of optimal timeslots, if any

    # Zeros mean completely unavailable, so remove them
    unavailable = np.where(availability == 0, 0, 1)
    unavailable = np.prod( unavailable, axis=0)
    availability *= unavailable

    time_score = np.prod(availability, axis=0)
    best_times = []

    max_time_score = np.max(time_score)
    if(max_time_score > 0):

        best_times = np.where(time_score == max_time_score)
        #best_times = best_times.to_list()
        best_times = best_times[0]

        best_times = [int(best_times[t]) for t in range(len(best_times))]


    return best_times

def doodlePairwise(availability):

    # Input: 2d matrix team*timeslot with timeslot preference weights
    # Output: 2d matrix, team*team, 1 if teams have a time they can meet, 0 if otherwise

    pairwise_availability = np.zeros((availability.shape[0], availability.shape[0]))
    suggested_times = np.zeros((availability.shape[0], availability.shape[0]), dtype=int)
    for i in range(availability.shape[0]):
        for j in range(i + 1, availability.shape[0]):
            available_both = np.concatenate(([availability[i]], [availability[j]]), axis=0)

            doodle_times = doodle(available_both)

            if len(doodle_times)> 0:
                pairwise_availability[i, j] = 1
                pairwise_availability[j, i] = 1
                suggested_time = int(np.random.choice(doodle_times))
                suggested_times[i, j] = suggested_time
                suggested_times[j, i] = suggested_time
                availability[i][suggested_time] = 0
                availability[j][suggested_time] = 0

 

    return pairwise_availability, suggested_times




def updateAvailability(pairwise_availability, matches):

    # Updates availability matrix to not repeat matches

    for i in range(matches.shape[0]):
        
        match = matches[i]
        if match >= 0:
            pairwise_availability[i, match] = 0
            pairwise_availability[match, i] = 0
    return pairwise_availability


    

def domainAvailability(teams_df):

    # 2d matrix: 1 if teams are in the same domain, 0 if otherwise

    pairwise_availability = np.zeros((teams_df.shape[0], teams_df.shape[0]))
    domains = teams_df['Domain'].unique()

    for domain in domains:
        domain_team = teams_df.loc[teams_df['Domain'] == domain]

        n_domain_team = len(domain_team)

        for i in range(n_domain_team):
            team1 = domain_team.index[i]

            for j in range(i + 1, n_domain_team):
                team2 = domain_team.index[j]
                pairwise_availability[ team1,  team2] = 1
                pairwise_availability[team2, team1] = 1


    return pairwise_availability


def nonDomainAvailability(teams_df):

    # 2d matrix: 1 if not in same domain, 0 otherwise 

    domain_availability = domainAvailability(teams_df)
    non_domain_availability = np.where(domain_availability == 1, 0, 1)
    # remove center line (A = A)
    for i in range(non_domain_availability.shape[0]):
        non_domain_availability[i, i] = 0
    return non_domain_availability
        
                

def matchPairs(pairwise_availability, already_matched = None):

    # Input: 2d pairwise availability (team*team)
    # Output: 1d matches: match[team1] = team2

    max_possible_score = pairwise_availability.shape[0] // 2
    current_best_score = 0
    current_output = np.full((pairwise_availability.shape[0]), -1)
    if already_matched is not None:
        already_matched = np.array(already_matched)

    
    for _ in range(1000):
        output = np.full((pairwise_availability.shape[0]), -1)
        my_av = pairwise_availability.copy()
        score = 0

        order = np.random.permutation(my_av.shape[0]).tolist()
        order = [int(o) for o in order]
        for i in order:

            possibilities = np.array(np.where(my_av[i] == 1))[0]

            if possibilities.shape[0] > 0:
                choice = np.random.choice(possibilities)

                output[i] = choice
                output[choice] = i

                score += 1

                my_av[:, i] = 0
                my_av[i,:] = 0
                my_av[:, choice] = 0
                my_av[choice,:] = 0

        # Penalty for matching with None if it has a previous None
        if already_matched is not None:

            for i in range(output.shape[0]):
                if -1 in already_matched[:, i] and output[i] == -1:
                    n_nones = sum(already_matched[:, i] == -1)
                    score -= 2*n_nones

        if score > current_best_score:
            current_best_score = score
            current_output = output
        if current_best_score >= max_possible_score:
            break

    

    return current_output, 2*current_best_score
    

def networkingMeetings(n_teams, n_meetings, n_per_meeting):
    total_slots = n_per_meeting*n_meetings
    n_to_cover = 1 + total_slots // n_teams

    rng = np.random.default_rng()
    
    teams = np.arange(n_teams)
    result = []
    for _ in range(n_meetings):
        rng.shuffle(teams)
        meeting = teams[:n_per_meeting]
        result.append(list(meeting))


    return result

    

def main(teams_filename, time_of_day_constraints_filename, main_schedule_filename):

    # load data
    teams_df = pd.read_csv(teams_filename)
    tod_constraints_df = pd.read_csv(time_of_day_constraints_filename)
    main_schedule = pd.read_csv(main_schedule_filename)
    main_schedule['Datetime'] = pd.to_datetime(main_schedule.Date + " " + main_schedule.Time)
    main_schedule.drop(columns=['Date', 'Time', 'Session', 'Notes'], inplace=True)

    # Networking meetings: Teams send a representative to each one, so can overlap
    n_network_sessions = 2
    network_sessions = []
    n_teams = teams_df.shape[0]
    n_meetings = 5
    n_per_meeting = 6
    for i in range(n_network_sessions):
        network_sessions.append(networkingMeetings(n_teams, n_meetings, n_per_meeting))
    

    # Given a csv with Teams, Domain, and Timezone, create a schedule 

    tod_constraints = tod_constraints_df['Availability'].to_numpy()
    tod_constraints = np.tile(tod_constraints, 3)

    initial_hour = main_schedule.Datetime.iloc[0].hour
    n_timeslots = main_schedule.shape[0]

    slots_per_hour = 2 # Half hour timeslots

    basic_availability = initAvailability(teams_df, main_schedule, tod_constraints, initial_hour, n_timeslots, slots_per_hour)


    doodle_pairwise_availability, suggested_times = doodlePairwise(basic_availability)



    # Use matching domains
    domain_availability = domainAvailability(teams_df)
    # Multiplication creates the constraint (zero if either are zero)
    pairwise_availability = doodle_pairwise_availability*domain_availability
   
    all_matches = []

    # First meeting (matched domain)
    matches1, score1 = matchPairs(pairwise_availability)



    print('match 1 score: ', score1)
    all_matches.append(matches1)


    # # Second meeting (matched domain)
    pairwise_availability = updateAvailability(pairwise_availability, matches1)
    matches2, score2 = matchPairs(pairwise_availability, all_matches)
    print('match 2 score: ', score2)
    all_matches.append(matches2)


    # # Third meeting (non-matching domains)
    pairwise_availability = updateAvailability(pairwise_availability, matches2)
    non_domain_availability = nonDomainAvailability(teams_df)
    
    # we can start over because inherently no matching domain pair can be a non-matching pair
    pairwise_availability = doodle_pairwise_availability*non_domain_availability
    matches3, score3 = matchPairs(pairwise_availability, all_matches)
    print('match 3 score: ', score3)
    all_matches.append(matches3)

    # # Fourth meeting (non-matching domains)
    pairwise_availability = updateAvailability(pairwise_availability, matches3)
    matches4, score4 = matchPairs(pairwise_availability, all_matches)
    print('match 4 score: ', score4)
    all_matches.append(matches4)

    # Output all
    all_matches = np.array(all_matches)
    printMatches(all_matches, teams_df)
    exportMatches(all_matches, suggested_times, teams_df, tod_constraints_df, main_schedule, 'teams_schedule.csv')
    exportFullSchedule(main_schedule, teams_df, all_matches, suggested_times, 'full_schedule.csv')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", type=str, default="teams.csv", help="Input teams spreadsheet")
    parser.add_argument("--tod", type=str, default="time_of_day_constraints.csv", help="Input time of day weights")
    parser.add_argument("--main", type=str, default="init_schedule.csv", help="Input main events schedule")
    args = parser.parse_args()

    teams_filename = args.teams
    tod_filename = args.tod
    main_filename = args.main
    main(teams_filename, tod_filename, main_filename)
