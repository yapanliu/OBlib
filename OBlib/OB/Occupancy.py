import pandas as pd


class Occupancy:
    def ashrae(self, x, sun_thursday, friday, saturday):
        ashrae_profile = pd.DataFrame({
            'Monday': sun_thursday,
            'Tuesday': sun_thursday,
            'Wednesday': sun_thursday,
            'Thursday': sun_thursday,
            'Friday': friday,
            'Saturday': saturday,
            'Sunday': sun_thursday,
        })

        for i in x.T:
            day_values = ashrae_profile[x.Date_Time.iloc[i].day_name()][0]
            hour = x.Date_Time.iloc[i].hour
            for j in range(len(day_values) - 1):
                start_hour = day_values[j][0]
                end_hour = day_values[j + 1][0]
                start_value = day_values[j][1]
                if (hour >= start_hour) and (hour < end_hour):
                    x.at[i, 'prediction'] = start_value
        return x

    def occupancy(self, x):
        # convert strings to datetime if needed
        if x.Date_Time.dtype == 'O':
            x['Date_Time'] = pd.to_datetime(x.Date_Time)

        x['prediction'] = ''

        # ASHRAE 90.1 occupancy profile, binary values
        sun_thursday = [[[0, 0], [6, 1], [24, 0]]]
        friday = [[[0, 0], [6, 1], [19, 0], [24, 0]]]
        saturday = [[[0, 0], [6, 1], [18, 0], [24, 0]]]

        Occupancy.ashrae(x, sun_thursday, friday, saturday)

        return x['prediction']

    def occupant_count(self, x):
        # convert strings to datetime if needed
        if x.Date_Time.dtype == 'O':
            x['Date_Time'] = pd.to_datetime(x.Date_Time)

        x['prediction'] = ''

        # ASHRAE 90.1 occupancy profile
        sun_thursday = [[[0, 0], [6, 0.1], [7, 0.2], [8, 0.95], [12, 0.5], [13, 0.95], [17, 0.3], [18, 0.1], [22, 0.05],
                         [24, 0]]]
        friday = [[[0, 0], [6, 0.1], [8, 0.3], [12, 0.1], [17, 0.05], [19, 0], [24, 0]]]
        saturday = [[[0, 0], [6, 0.05], [18, 0], [24, 0]]]

        Occupancy.ashrae(x, sun_thursday, friday, saturday)

        return x['prediction']
