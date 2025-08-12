import pandas as pd
from typing import Dict, Union
import numpy as np

class ESGMetricsCalculator:
    '''
    A class to calculate various ESG metrics including carbon footprint, renewable energy, waste management, labor practices, diversity and inclusion, community engagement, board diversity, executive pay, anti-corruption policies, and environmental scores.
    '''

    def __init__(self):
        self.total_esg_score = {} # adding a private variable to store the total ESG score

    def calculate_weighted_carbon_footprint_score(self, score, weight):
        '''
        Takes the score of each carbon footprint category and the weight corresponding to each category and returns the carbon footprint weighted score.
        Inputs:
            score(float): A float number of the score for each carbon footprint category
            weight(float): A float number of the weight of each carbon footprint category
        Returns:
            float: A float number representing the weighted carbon footprint score
        '''
        weighted_carbon_footprint_score = score * weight
        return weighted_carbon_footprint_score


    def calculate_carbon_footprint_score(self, carbon_footprint_data: dict, weights: dict):
        '''
        Takes a carbon footprint data dictionary and a dictionary of weights corresponding to each carbon footprint category and returns the total carbon footprint score.
        Inputs:
            carbon_footprint_data(dict): A dictionary of carbon footprint data
            weights(dict): A dictionary of weights for each category
        Returns:
            float: A float number of the total carbon footprint score
        '''
        total_carbon_footprint_score = sum(
            self.calculate_weighted_carbon_footprint_score(carbon_footprint_data[category], weights[category]) for category
            in carbon_footprint_data)
        return total_carbon_footprint_score


    def calculate_carbon_footprint(self, activity_data, emission_factor):
        '''
            Calculates the carbon footprint by measuring green house gas emissions produced directly and indirectly by a company.
            Inputs:
                activity_data(dictionary): A dictionary of activity data
                emission_factor(float): A float number of emission factor
            Returns:
                float: A float number of green house gas emissions
        '''
        ghg_emissions = activity_data * emission_factor
        return ghg_emissions


    def calculate_weighted_renewable_energy_score(self, score, weight):
        '''
        Takes a score of each renewable energy category and a weight corresponding to each renewable energy category and returns the weighted score.
        Inputs:
            score(float): A float number of the score for each renewable energy category
            weight(float): A float number of the weight of each renewable energy category
        Returns:
             float: A float number representing the weighted renewable energy score
        '''
        weighted_renewable_energy_score = score * weight
        return weighted_renewable_energy_score


    def calculate_renewable_energy_score(self, renewable_energy_data: dict, weights: dict):
        '''
        Takes the renewable energy data dictionary and the dictionary of weights corresponding to the renewable energy dictionary and returns the total renewable energy score.
        Inputs:
            renewable_energy_data(dict): A dictionary of renewable energy data
            weights(dict): A dictionary of weights for each category
        Returns:
             float: A float number representing the total renewable energy score
        '''
        total_reneweable_energy_score = sum(
            self.calculate_weighted_renewable_energy_score(renewable_energy_data[category], weights[category]) for category
            in renewable_energy_data)
        return total_reneweable_energy_score


    def calculate_energy_consumption(self, energy_data: dict) -> pd.DataFrame:
        ''' Takes the energy consumption data and returns the total energy consumption for an organisation.
        Inputs:
            energy_data(dict): A dictionary of energy consumption data
        Returns:
            pd.DataFrame: A DataFrame containing the total energy consumption of an organisation
        '''

        # Create a DataFrame for the total energy consumption data
        df = pd.DataFrame(energy_data)
        df['Total Energy Consumption (kWh'] = df.sum(axis=1)

        # Calculating annual total energy consumption
        annual_total_consumption = df['Total Energy Consumption (kWh)'].sum()

        # Adding annual total to the DataFrame
        df.loc['Annual Total'] = df.sum()
        df.at[
            'Annual Total', 'Total Energy Consumption (kWh)'] = annual_total_consumption  # Locating the annual total consumption

        return df


    def calculate_water_usage(self, water_usage_data: dict) -> pd.DataFrame:
        '''
        Takes the water usage data and returns the total water usage for an organisation.
        Inputs:
            water_usage_data(dict): A dictionary containing water sources and values as lists of monthly usage
        Returns:
            pd.DataFrame: A DataFrame containing the total water usage for each month
        '''
        water_usage_df = pd.DataFrame(water_usage_data)
        # Calculating total water usage for each month
        water_usage_df['Total Water Usage (cubic meters)'] = water_usage_df.sum(axis=1)

        # Calculating annual total water usage
        annual_total_water_usage = water_usage_df['Total Water Usage (cubic meters)'].sum()

        # Adding annual total for the DataFrame
        water_usage_df.loc['Annual Total'] = water_usage_df.sum()
        water_usage_df.at['Annual Total', 'Total Water Usage (cubic meters)'] = annual_total_water_usage

        return water_usage_df


    def calculate_weighted_waste_score(self, quantity, score):
        '''
        Takes the quantity and score and calculates the weighted score for each waste category.
        Inputs:
            quantity(float): A float number representing the quantity of waste category
            score(float): A float number representing the weighted score
        Returns:
            float: A float number representing the weighted score
        '''
        weighted_score = quantity * score
        return weighted_score


    def calculate_waste_management(self, waste_data: dict) -> pd.DataFrame:
        '''
        Takes the waste data and returns the total waste management score for an organisation.
        Inputs:
            waste_data(dict): A dictionary containing waste data
        Returns:
            pd.DataFrame: A DataFrame containing the total waste management for an organisation
        '''
        total_quantity = sum(quantity for quantity, _ in waste_data.values())  # ignoring the scores
        total_weighted_score = sum(
            self.calculate_weighted_waste_score(quantity, score) for quantity, score in waste_data.values())
        waste_management_score = pd.DataFrame([{'Waste Management Score': total_weighted_score / total_quantity}]) if total_quantity else pd.DataFrame([{'Waste Management Score': 0}])
        return waste_management_score


    def weighted_labor_score(self, score, weight):
        '''
        Takes the score of each category and the corresponding weight to calculate the weighted labor score.
        Inputs:
            score(float): A float number representing the score for each labor pratice
            weight(float): A float number representing the weight of each labor pratice
        Returns:
            float: A float number representing the weighted labor score
        '''
        labor_weighted_score = score * weight
        return labor_weighted_score


    def calculate_labor_practices_score(self, labor_data: dict, weights: dict):
        '''
        Takes the labor data, score and weight and returns the total labor practices score for an organisation.
        Inputs:
            labor_data(dict): A dictionary containing labor practice data
            weights(dict): A dictionary containing weights corresponding to categories of labor practices
        Returns:
            float: A float number of total labor practices score
        '''
        total_labor_practices_score = sum(
            self.calculated_weighted_labor_score(labor_data[category], weights[category]) for category in labor_data)
        return total_labor_practices_score


    def calculate_di_weighted_score(self, score, weight):
        '''
        Takes the score of each category and the corresponding weight to calculate the total diversity and inclusion score.
        Inputs:
            score(float): A float number representing the score for each D&I category
            weight(float): A float number representing the weight of each D&I category
        Returns:
            float: A float number representing the total diversity and inclusion score
        '''
        d_and_i_score = score * weight
        return d_and_i_score


    def calculate_di_score(self, di_data: dict, weights: dict):
        '''
        Takes the diversity and inclusion score for each category and the corresponding weight for each category to calculate the total diversity and inclusion score.
        Inputs:
            di_data(dict): A dictionary containing a diversity and inclusion score for each category
            weights(dict): A dictionary containing a weight for each category
        Returns:
            float: A float number representing the total diversity and inclusion score
        '''
        total_di_score = sum(
            self.calculate_diversity_and_inclusion_weighted_score(di_data[category], weights[category]) for category in
            di_data)
        return total_di_score


    def calculate_weighted_ce_score(self, score, weight):
        '''
        Takes the score of each community engagement category and the weight for each and returns the weighted score.
        Inputs:
            score(float): A float number representing the score for each community engagement category
            weight(float): A float number representing the weight of each community engagement category
        Returns:
            float: A float number representing the weighted score
        '''
        weighted_ce_score = score * weight
        return weighted_ce_score


    def calculate_ce_score(self, ce_data: dict, weights: dict):
        '''
        Takes a dictionary of community engagement data and the weights corresponding to community engagement categories and returns the total community engagement score.
        Inputs:
            ce_data(dict): A dictionary containing community engagement data
            weights(dict): A dictionary containing a weight for each community engagement category
        Returns:
            float: A float number representing the total community engagement score
        '''
        ce_score = sum(self.calculate_weighted_ce_score(ce_data[category], weights[category]) for category in ce_data)
        return ce_score


    def calculate_weighted_bd_score(self, score, weight):
        '''
        Takes the board diversity score for each category and the weight for each and returns the weighted board diversity score.
        Inputs:
            score(float): A float number representing the score for each board diversity category
            weight(float): A float number representing the weight of each board diversity category
        Returns:
            float: A float number representing the weighted board diversity score
        '''
        weighted_bd_score = score * weight
        return weighted_bd_score


    def calculate_bd_score(self, bd_data: dict, weights: dict):
        '''
        Takes a dictionary of board diversity data and the weights corresponding to board diversity categories and returns the total board diversity score.
        Inputs:
            bd_data(dict): A dictionary containing board diversity data
            weights(dict): A dictionary containing a weight for each board diversity category
        Returns:
            float: A float number representing the total board diversity score
        '''
        total_db_score = sum(self.calculate_weighted_bd_score(bd_data[category], weights[category]) for category in bd_data)
        return total_db_score


    def calculate_weighted_ep_score(self, score, weight):
        '''
        Takes the score of each executive pay category and the weight for each and returns the weighted score.
        Inputs:
            score(float): A float number representing the score for each executive pay category
            weight(float): A float number representing the weight of each executive pay category
        Returns:
            float: A float number representing the weighted score
        '''
        weighted_ep_score = score * weight
        return weighted_ep_score


    def calculate_ep_score(self, ep_data: dict, weights: dict):
        '''
        Takes a dictionary containing the executive pay score for each category and the dictionary of weights corresponding to the categories and returns the total score.
        Inputs:
            ep_data(dict): A dictionary of executive pay data.
            weights(dict): A dictionary containing a weight for each executive pay category
        Returns:
            float: A float number representing the weighted score
        '''
        total_ep_score = sum(self.calculate_weighted_ep_score(ep_data[category], weights[category]) for category in ep_data)
        return total_ep_score


    def calculate_weighted_ap_score(self, score, weight):
        '''
        Takes the score of anti-corruption policy score for each category and the weight for each and returns the weighted score.
        Inputs:
            score(float): A float number of the score for each anti-corruption policies category
            weight(float): A float number of the weight for each anti-corruption policies category
        Returns:
            float: A float number representing the weighted anti-corruption policies score
        '''
        weighted_ap_score = score * weight
        return weighted_ap_score


    def calculate_ap_score(self, ap_data: dict, weights: dict):
        '''
        Takes a dictionary of anti-corruption data and the dictionary of weights for anti-corruption categories and returns the total score.
        Inputs:
            ap_data(dict): A dictionary of anti-corruption policies data
            weights(dict): A dictionary of weights corresponding to anti-corruption policies categories
        Returns:
            float: A float number representing the total anti-corruption policies score
        '''
        total_ap_score = sum(self.calculate_weighted_ap_score(ap_data[category], weights[category]) for category in ap_data)
        return total_ap_score


    def calculate_weighted_environmental_score(self, score, weight):
        '''
        Takes the score of each environmental category and the weight for each and returns the weighted environmental score.
        Inputs:
            score(float): A float number representing the score for each environmental category
            weight(float): A float number representing the weight of each environmental category
        Returns:
            float: A float number representing the weighted environmental score
        '''
        weighted_environmental_score = score * weight
        return weighted_environmental_score


    def calculate_environmental_score(self, environmental_data: dict, weights: dict):
        '''
        Takes the environmental data dictionary and the dictionary of weights corresponding to environmental categories and returns the total environmental score.
        Inputs:
            environmental_data(dict): A dictionary containing environmental data
            weights(dict): A dictionary containing a weight for each environmental category
        Returns:
             float: A float number representing the total environmental score
        '''
        total_environmental_score = sum(
            [self.calculate_carbon_footprint_score.total_carbon_footprint_score + self.calculate_renewable_energy_score.total_renewable_energy_score + self.calculate_waste_management.total_waste_management_score])
        return total_environmental_score


    def calculate_social_score(self, environmental_data: dict, weights: dict):
        '''
        Takes the environmental data dictionary and the dictionary of weights corresponding to environmental categories and returns the total social score.
        Inputs:
            environmental_data(dict): A dictionary containing environmental data
            weights(dict): A dictionary containing a weight for each environmental category
        Returns:
            float: A float number representing the total social score
        '''
        total_social_score = sum(
            [self.calculate_labor_practices_score.total_labor_practices_score + self.calculate_di_score.total_di_score + self.calculate_ce_score.total_ce_score])
        return total_social_score


    def calculate_governance_score(self, governance_data: dict, weights: dict):
        '''
        Takes the governance data dictionary and the dictionary of weights corresponding to governance categories and returns the total governance score.
        Inputs:
            governance_data(dict): A dictionary containing governance data
            weights(dict): A dictionary containing weights for each governance category
        Returns:
            float: A float number representing the total governance score
        '''
        total_governance_score = sum(
            [self.calculate_ap_score.total_ap_score + self.calculate_bd_score.total_bd_score + self.calculate_ep_score.total_ep_score])
        return total_governance_score

    def calculate_esg_score(self, environmental_data: dict, social_data: dict, governance_data: dict, weights: dict) -> float:
        '''
        Takes the environmental, social and governance data dictionaries and the dictionary of weights corresponding to each category and returns the total ESG score.
        Inputs:
            environmental_data(dict): A dictionary containing environmental data
            social_data(dict): A dictionary containing social data
            governance_data(dict): A dictionary containing governance data
            weights(dict): A dictionary containing weights for each category
        Returns:
            float: A float number representing the total ESG score
        '''
        total_environmental_score = self.calculate_environmental_score(environmental_data, weights['environmental'])
        total_social_score = self.calculate_social_score(social_data, weights['social'])
        total_governance_score = self.calculate_governance_score(governance_data, weights['governance'])

        total_esg_score = total_environmental_score + total_social_score + total_governance_score
        return total_esg_score

def calculate_biodiversity_units(
        area: float,
        distinctiveness: float,
        condition: float,
        strategic_significance: float,
        connectivity: float,
) -> float:
    '''
    Calculates the biodiversity units based on area, distinctiveness, condition, strategic significance, and connectivity.
    Args:
        area (float): Area of the habitat in hectares.
        distinctiveness (float): Distinctiveness score of the habitat (0-1).
        condition (float): Condition score of the habitat (0-1).
        strategic_significance (float): Strategic significance score of the habitat (0-1).
        connectivity (float): Connectivity score of the habitat (0-1).
    Returns:
        float: Calculated biodiversity units.
    '''
    return area * distinctiveness * condition * strategic_significance * connectivity

def calculate_species_richness(
        total_species: int,
        area: float,
        strict: bool = True,
) -> Union[int, float]:
    '''
    Calculates the species richness based on the total number of species and the area.
    Args:
        total_species (int): Total number of species in the area.
        area (float): Area of the habitat in hectares.
    Returns:
        Union[int, float]: Calculated species richness (species per hectare).
    '''
    if area <= 0:
        if strict:
            raise ValueError("Area must be greater than zero for strict mode.")
        else:
            # In non-strict mode, return NaN or zero to indicate invalid calculation
            return float('nan')

    return total_species / area

def calculate_shannon_wiener_index(
        n_i: int,
        N: int,
        strict: bool = True
)-> float:
    '''
    Calculates the Shannon-Wiener index (more sensitive to rare species, capturing subtle shifts, mirroring entropy-based reasoning) based on the number of individuals of each species (n_i) and the total number of individuals (N).
    Args:
        n_i (int): Number of individuals of a species.
        N (int): Total number of individuals in the community.
    Returns:
        float: Calculated Shannon-Wiener index.
    '''
    if N <= 0:
        if strict:
            raise ValueError("Total number of individuals (N) must be greater than zero.")
        else:
            # In non-strict mode, return NaN or zero to indicate invalid calculation
            return float('nan')

    proportion = n_i / N
    log = np.log(proportion) if proportion > 0 else np.nan  # Avoid log(0)
    return -proportion * log

#def calculate_habitat_condition_score(

#)

