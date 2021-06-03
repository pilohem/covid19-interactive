from bokeh.models.annotations import Legend
import pandas as pd
import numpy as np

df = pd.read_csv('covid-data-2021-05-24.csv')
df['date'] = pd.to_datetime(df['date'])

df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

df['new_cases_density'] = df['new_cases'] / df['population']*100

df['total_vaccinations_density'] = df['total_vaccinations'] / df['population']*100

df_gp = df.groupby(['year', 'month', 'iso_code', 'continent'])

df_gp_mean = df_gp[['total_vaccinations_density', 'new_cases_density']].mean().reset_index()

from bokeh.io import curdoc, output_notebook, reset_output
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, Slider, Select,MultiSelect, Div
from bokeh.layouts import widgetbox, row, column
from bokeh.palettes import Category20_20

# output_notebook()

countries_list = df_gp_mean.iso_code.unique().tolist()
continents = df_gp_mean.continent.unique().tolist()

desc = Div(text='Div', sizing_mode="stretch_width")

# reset_output()
color_mapper = CategoricalColorMapper(factors=countries_list, palette=Category20_20)

source = ColumnDataSource(data=dict(x=[], y=[], color=[], month=[], year=[]))

# Create Input controls
slider_year = Slider(start=min(df_gp_mean.year), end=max(df_gp_mean.year),
               step=1, value=min(df_gp_mean.year), title='Year')
slider_month = Slider(start=min(df_gp_mean.month), end=max(df_gp_mean.month),
               step=1, value=min(df_gp_mean.month), title='Month')
select_continent = Select(title="Continent", options=sorted(continents), value="North America")
select_countries = MultiSelect(value=['MEX', 'USA', 'CAN'], title='Countries', options=sorted(countries_list))


def select_data():
    print('SELECT RUNNING')
    df_selected = df_gp_mean[
        (df_gp_mean['year'] >= slider_year.value) &
        (df_gp_mean['month'] >= slider_month.value) &
        (df_gp_mean['continent'] == select_continent.value)]
    return df_selected


def filter_countries():
    print('FILTER RUNNING')
    df_year_month_conti = select_data()
    selected_all = pd.DataFrame()
    for c in select_countries.value:
        selected_c = df_year_month_conti[df_year_month_conti['iso_code'] == c]
        selected_all = selected_all.append(selected_c)
    return selected_all


def update_plot():
    print('UPDATE RUNNING')
    df = filter_countries()
    print(df.shape)
    source.data = dict(
        x=df['new_cases_density'],
        y=df['total_vaccinations_density'],
        color=df['iso_code'],
        month=df["month"],
        year=df["year"],
    )
    # print(source.data['color'])


controls = [slider_year, slider_month, select_continent,  select_countries]
select_continent.on_change('value', lambda attr,  old, new: update_plot())
for control in controls:
    control.on_change('value', lambda attr, old, new: update_plot())

p = figure(title="Covid19 in the World", sizing_mode="scale_both",
            plot_width=350, plot_height=200)
p.circle(x="x", y="y", source=source, size=10)

inputs = column(*controls)
l = column(desc, row(inputs, p), sizing_mode="scale_both")
update_plot()
curdoc().add_root(l)
curdoc().title = 'Covid19'