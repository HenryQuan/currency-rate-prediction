"""
Convert raw json data to csv for learning
"""

import json
import os

IO_NAME = 'AUD2CNY'
if __name__ == '__main__':
    file_path = '../raw/{}.json'.format(IO_NAME)
    if not os.path.exists(file_path):
        exit('File not found: {}'.format(file_path)v)

    with open(file_path, 'r') as raw_data:
        data = json.load(raw_data)
        ten_years = data['batchList'][0]
        rates = ten_years['rates'][1:]
        with open('{}.csv'.format(IO_NAME), 'w') as csv:
            csv.write('Day,Rate\n')
            for index, rate in enumerate(rates):
                csv.write('{},{}\n'.format(index, rate))

        raw_data.close()
