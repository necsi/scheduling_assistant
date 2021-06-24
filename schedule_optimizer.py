from datetime import time
from os import pardir
from numpy.core.numerictypes import maximum_sctype
from ortools.sat.python.cp_model import Domain
import pandas as pd
import numpy as np
import csv


def printMatches(all_matches, team_df):

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

def exportMatches(all_matches, team_df, filename='matches.csv'):
    headers = ['Team', 'Domain Match 1', 'Domain Match 2', 'Non-domain Match 1', 'Non-domain Match 2']
    n_teams = team_df.shape[0]
    all_matches = all_matches.T


    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for t in range(n_teams):
            row = [team_df['Team'][t]]
            for m in range(all_matches[t].shape[0]):
                t2 = int(all_matches[t, m])
                if t2 >= 0:
                    row.append(team_df['Team'][t2])
                else:
                    row.append('None')
            writer.writerow(row)
    print('wrote file', filename)
        


def initAvailability(teams_df, basic):
    availability = []
    timezones = teams_df['Timezone'].to_list()
    for tz in timezones:
        availability.append(np.roll(basic, tz))
    return np.array(availability)

def updateAvailability(availability, matches):
    for i in range(matches.shape[0]):
        
        match = matches[i]
        if match >= 0:
            availability[i, match] = 0
            availability[match, i] = 0
    return availability

def domainAvailability(teams_df):
    availability = np.zeros((teams_df.shape[0], teams_df.shape[0]))
    domains = teams_df['Domain'].unique()

    for domain in domains:
        domain_team = teams_df.loc[teams_df['Domain'] == domain]

        n_domain_team = len(domain_team)

        for i in range(n_domain_team):
            team1 = domain_team.index[i]

            for j in range(i + 1, n_domain_team):
                team2 = domain_team.index[j]
                availability[ team1,  team2] = 1
                availability[team2, team1] = 1


    return availability



def doodle(availability):

    # Zeros mean completely unavailable, so remove them
    unavailable = np.where(availability == 0, 0, 1)
    unavailable = np.prod( unavailable, axis=0)

    availability *= unavailable

    # Use availability time scores to find optimal times
    time_score = np.prod(availability, axis=0)

    best_times = np.where(time_score == np.max(time_score), 1, 0)

    return best_times

def doodlePairwise(availability):
    pairwiseAvailability = np.zeros((availability.shape[0], availability.shape[0]))
    for i in range(availability.shape[0]):
        for j in range(availability.shape[0]):
            a = np.concatenate(([availability[i]], [availability[j]]), axis=0)

            doodle_times = doodle(a)

 
            pairwiseAvailability[i, j] = np.max(doodle_times) # 1 if a time is available, 0 othrewise

    return pairwiseAvailability


                

def matchPairs(availability):
    max_possible_score = availability.shape[0] // 2
    current_best_score = 0
    current_output = np.ndarray((0))
    
    for _ in range(1000):
        output = np.full((availability.shape[0]), -1)
        my_av = availability.copy()
        score = 0
        # print()
        # print('overall ', _)
        order = np.random.permutation(my_av.shape[0]).tolist()
        order = [int(o) for o in order]
        for i in order:
            #print('i', i)
            #print(my_av)
            possibilities = np.array(np.where(my_av[i] == 1))[0]
            #print('possibilities', possibilities)

            if possibilities.shape[0] > 0:
                choice = np.random.choice(possibilities)
                output[i] = choice
                output[choice] = i

                score += 1

                my_av[:, i] = 0
                my_av[i,:] = 0
                my_av[:, choice] = 0
                my_av[choice,:] = 0
                #print('choice', choice)
             
        if score > current_best_score:
            current_best_score = score
            current_output = output
        if current_best_score >= max_possible_score:
            break

    return current_output, 2*current_best_score
    
            
            


def main():



    teams = pd.read_csv('teams.csv')



    basic_availability = np.array([
        0, # 12 am
        0, 
        0, 
        0, 
        0, 
        0, 
        0, 
        1, # 7 am
        2, 
        3, # 9 am
        3, 
        3, 
        3, #12 pm
        3, 
        3, 
        3, 
        3, 
        3, # 5pm
        2, 
        2, 
        2, 
        1, # 9 pm
        0, 
        0

    ])




    # Team matches: use matching domains
    availability = initAvailability(teams, basic_availability)
    availability = doodlePairwise(availability)
    domain_availability = domainAvailability(teams)
    availability *= domain_availability

    # first meeting
    matches1, score1 = matchPairs(availability)

    # remove matches from next round
    availability = updateAvailability(availability, matches1)

    #second meeting
    matches2, score2 = matchPairs(availability)


    availability = updateAvailability(availability, matches2)


    # Non team matches: use non-matching domains
    non_domain_availability = np.where(domain_availability == 1, 0, 1)
    # remove center line (A = A)
    for i in range(non_domain_availability.shape[0]):
        non_domain_availability[i, i] = 0
    availability = initAvailability(teams, basic_availability)
    availability = doodlePairwise(availability)
    availability *= non_domain_availability

    # third meeting
    matches3, score3 = matchPairs(availability)
    availability = updateAvailability(availability, matches3)

    # fourth meeting
    matches4, score4 = matchPairs(availability)

    all_matches = np.array([matches1, matches2, matches3, matches4])
    printMatches(all_matches, teams)
    exportMatches(all_matches, teams, 'schedule.csv')





if __name__ == '__main__':
    main()
