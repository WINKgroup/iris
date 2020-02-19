from bokeh.sampledata.iris import flowers as df
from bokeh.plotting import figure, output_file, show
from bokeh.models import CategoricalColorMapper
from bokeh.palettes import Category10 as Category10

mapper = CategoricalColorMapper(factors=['setosa', 'virginica', 'versicolor'], palette=Category10[3])
plot = figure(x_axis_label='petalo', y_axis_label="sepalo")
plot.circle('petal_length', 'sepal_length', size=10, source=df, color={'field': 'species', 'transform': mapper}, legend='species')
show(plot)