import json
import os
from re import match
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as figfact
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

colors = [#default plotly colors
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
    ]
line_dash_shape = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
class GeneratePlots:
    def __init__(self, experiments):
        self.experiments = experiments

    def generateDensityRidge(self, key, across_cols=True):
        # do it across columns
        if across_cols:
            match = key
        data = {
            'focused_feature': [],
            'cell_activations': [],
            'sensation': []
        }
        for ex in self.experiments:
            for obj in ex['results']:
                if across_cols:
                    keys = [key for key in obj if match in key]
                    for key in keys:
                        for isensation, sensation in enumerate(obj[key]):
                            data['focused_feature'].append(ex['params']['experiment'][ex['focused_feature']])
                            data['cell_activations'].append(sensation)
                            data['sensation'].append(isensation)
                else:
                    for isensation, sensation in enumerate(obj[key]):
                        data['focused_feature'].append(ex['params']['experiment'][ex['focused_feature']])
                        data['cell_activations'].append(sensation)
                        data['sensation'].append(isensation)

        df = pd.DataFrame(data, columns=['focused_feature', 'cell_activations', 'sensation'])

        fig = go.Figure()
        for exp in self.experiments:
            fig.add_trace(go.Violin(y=df['sensation'][(df['focused_feature'] == exp['params']['experiment'][exp['focused_feature']])],# & (df['sensation'] != 0)],
                            x=df['cell_activations'][(df['focused_feature'] == exp['params']['experiment'][exp['focused_feature']])],# & (df['sensation'] != 0)],
                            name=exp['params']['experiment'][exp['focused_feature']],
                            legendgroup=exp['params']['experiment'][exp['focused_feature']])
                )
        fig.update_traces(orientation='h', side='negative', width=3, points=False)
        fig.update_layout(xaxis_zeroline=False)
        fig.update_layout(legend_title_text=exp['focused_feature'])
        fig.update_layout(title="Density of " + match if across_cols else key)
        fig.update_yaxes(range=[0,len(self.experiments[-1]['results'][0][key])], autorange="reversed")
        fig.update_xaxes(rangemode="tozero")
        return fig

    def generateDensity(self, key, yaxis_title='Cell Activations', normalization_key='Location', legend_title_text=None, show_title=True):
        """Generate density plot from arbitrary key. This could be any stream
        according to touches as an x-axis. Key can also be used to match other applicable
        keys, such as different columns

        Args:
            key (str): Key for data stream from results across experiments.
            yaxis_title (str) : Y-axis title override.

        Returns:
            fig (plotly.go): plotly graph to export to neptune or save to disk.
        """
        match = key
        percentiles = [25, 50, 75]
        data = []

        for iexp, exp in enumerate(self.experiments):
            if normalization_key == 'Location':
                # (cells x cells) x number of modules
                norm_divisor = (exp['params']['experiment']['cells_per_axis']**2)*exp['params']['experiment']['num_modules']
            data.append({
                'focused_feature' : exp['focused_feature'],
                exp['focused_feature'] : exp['params']['experiment'][exp['focused_feature']],
                'data' : [],
                'error_below' : [],
                'error_above': [],
            })
            # use num_sensations
            density = np.ndarray((0,exp['params']['experiment']['num_sensations']))
            for obj in exp['results']:
                # go over all cols
                keys = [key for key in obj if match in key]
                for key in keys:
                    density = np.vstack((density, np.array([obj[key]])/norm_divisor ))
            p1, p2, p3 = np.percentile(density, percentiles, axis=0)
            data[iexp]['data'] = p2.tolist()
            data[iexp]['error_below'] = (p2-p1).tolist()
            data[iexp]['error_above'] = (p3-p2).tolist()
        fig = go.Figure()
        for trace in data:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1,len(self.experiments[0]['results'][0][key])+1)),
                    y=trace['data'],
                    name = str(trace[trace['focused_feature']]),
                    mode = "lines+markers",
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=trace['error_above'],
                        arrayminus=trace['error_below']
                    )
                )
            )
        # fig.update_layout(legend=dict(x=0.62, y=0.99))
        if show_title:
            fig.update_layout(title="Density of " + match)
        if legend_title_text:
            fig.update_layout(legend_title_text=legend_title_text)
        else:
            fig.update_layout(legend_title_text=exp['focused_feature'])
        # fig.update_layout(template="plotly_dark")
        fig.update_layout({'xaxis': {'title': {'text': 'Number of Sensations'}}})
        fig.update_layout({'yaxis': {'title': {'text': yaxis_title}}})

        return fig

    def generateTouchesHist(self, title_text=None, legend_title_text=None, show_title=True, y_axis_title=None):
        data = {
            'touches': [],
            self.experiments[0]['focused_feature']: []
        }
        for exp in self.experiments:
            for obj in exp['results']:
                data['touches'].append(obj['touches'])
                data[exp['focused_feature']].append(exp['params']['experiment'][exp['focused_feature']])
        df = pd.DataFrame(data, columns=[exp['focused_feature'], 'touches'])
        fig = px.histogram(df, x="touches", color=exp['focused_feature'],
                            marginal="box", # or violin, rug
                            barmode='group',
                            histnorm='percent',
                            hover_data=df.columns)
        if title_text and show_title:
            fig.update_layout(title=title_text)
        elif show_title:
            fig.update_layout(title="Effect of \""  + exp['focused_feature'] + "\" on convergence")
        if legend_title_text:
            fig.update_layout(legend_title_text=legend_title_text)
        if y_axis_title:
            fig.update_layout({'yaxis': {'title': {'text': y_axis_title}}})
        else:
            fig.update_layout({'yaxis': {'title': {'text': 'Percentage'}}})
        fig.update_layout({'xaxis': {'title': {'text': 'Iteration Converged'}}})
        return fig

    def generateInferredDist(self, x_axis_title = None):
        data = {
            # self.experiments[0]['focused_feature']: [],
        }
        total = 0
        for exp in self.experiments:
            for obj in exp['results']:
                total += 1
                if obj['touches'] == None:
                    # data[exp['focused_feature']].append(exp['params']['experiment'][exp['focused_feature']])
                    data[str(exp['params']['experiment'][exp['focused_feature']])] = data.get(str(exp['params']['experiment'][exp['focused_feature']]), 0) + 1
                else:
                    # just add zero to make sure there is something
                    data[str(exp['params']['experiment'][exp['focused_feature']])] = data.get(str(exp['params']['experiment'][exp['focused_feature']]), 0) + 0
        # df = pd.DataFrame(data, columns=[exp['focused_feature']])
        # fig = px.histogram(df, x=exp['focused_feature'], color=exp['focused_feature'],
        #             # marginal="box", # or violin, rug
        #             # barmode='group',
        #             # histnorm='percent',
        #             hover_data=df.columns,
        #             title = "Effect of \""  + exp['focused_feature'] + "\" on inference error")
        fig = go.Figure(data=go.Scatter(x=[i for i in data.keys()], y=[float(i)/total for i in list(data.values())]))
        fig.update_layout(title="Effect of %s on inference error " % exp['focused_feature'])
        if x_axis_title:
            fig.update_layout({'xaxis': {'title': {'text': x_axis_title}}})
        else:
            fig.update_layout({'xaxis': {'title': {'text': exp['focused_feature']}}})
        fig.update_layout({'yaxis': {'title': {'text': 'Error'}}})
        return fig
    
    def generateDirectionalSelectivityStacked(self, direction):
        # wys plot vir 'n spesifieke bin in die ideal orientation (i.e. 0-36, 36-72, etc)
        data = {
            'direction': [],
            'firing_rate': [],
            'sensation': [],
            self.experiments[0]['focused_feature']: []
        }
        for exp in self.experiments:
            perModRange=exp['params']['experiment']['angle']/exp['params']['experiment']['num_modules']
            for obj in exp['results']:
                # only in specified direction and if algo was used
                if obj['ideal_orientation'] == direction and exp['params']['experiment']['orientationAlgo']==2:
                    for iorientation, orientation in enumerate(obj['orientational_firing_rate']):
                        for col in orientation:
                            for isensation, rate in enumerate(col):
                                # first make for individual cols then agrigated?
                                data['direction'].append(iorientation*perModRange)
                                if rate==0:
                                    data['firing_rate'].append(1/1000)
                                    print('somethin fishy here')
                                else:
                                    data['firing_rate'].append(rate)
                                data['sensation'].append(isensation+1)
                                data[exp['focused_feature']].append(exp['params']['experiment'][exp['focused_feature']])
                # else:
                #     print('It seems rondomness could suck sometimes, it seems it didnt generate an object at this rotation %d' % direction)
                #     return go.Figure()
        df = pd.DataFrame(data, columns=[exp['focused_feature'], 'direction', 'firing_rate', 'sensation'])
        # sum sensations in every direction
        # df = df.groupby(['direction', 'sensation'], as_index=False).sum()
        # fig = px.bar_polar(df, r="firing_rate", theta="direction",
        #                     color="sensation")
        # summ over all values in a direction
        df = df.groupby(['direction', exp['focused_feature']], as_index=False).sum()
        # normalize
        for feat in [p[0] for p in df.groupby(exp['focused_feature'])]:
            dd = df['firing_rate'][df[exp['focused_feature']]==feat]
            df['firing_rate'][df[exp['focused_feature']]==feat] = dd/max(dd)
        fig = px.line_polar(df, r="firing_rate", theta="direction", line_close=True,
                            range_r = [df.min()['firing_rate'], df.max()['firing_rate']],
                            color=exp['focused_feature'])
        fig.update_layout(title="Orientation selectivity of direction %d" % direction)
        return fig

    
    def generateDirectionalError(self, limit_directional_error=None, mag_legend_title_text=None, dir_legend_title_text=None, show_titles=True):
        data = {
            'chosen_direction': [],
            'direction': [],
            'error': [],
            self.experiments[0]['focused_feature']: []
        }
        for exp in self.experiments:
            perModRange=exp['params']['experiment']['angle']/exp['params']['experiment']['num_modules']
            for obj in exp['results']:
                ideal = obj['ideal_orientation']
                # somehow there are modules centred at 360... so im making them 0. this might be because of rounding up or something in the ideal case
                if obj['ideal_orientation'] == exp['params']['experiment']['num_modules']: # i think the only effect for this will be in plotting, since rotations will have the same effect
                    ideal = 0
                if obj['chosen_orientation'] == exp['params']['experiment']['num_modules']:
                    print('defuq 1')

                data['direction'].append(ideal*perModRange)
                data['chosen_direction'].append(obj['chosen_orientation']*perModRange)
                data['error'].append(np.absolute(ideal-obj['chosen_orientation'])*perModRange)
                data[exp['focused_feature']].append(exp['params']['experiment'][exp['focused_feature']])
        df = pd.DataFrame(data, columns=[exp['focused_feature'], 'direction', 'error', 'chosen_direction'])
        # fig = px.bar_polar(df, r="error", theta="direction", color_discrete_sequence= px.colors.sequential.Plasma_r,barmode='group',barnorm='percent',
        #                     color=exp['focused_feature'])

        # df = df.groupby(['direction', exp['focused_feature']], as_index=False).sum()
        # fig = px.line_polar(df, r="error", theta="direction", color=exp['focused_feature'], line_close=True)
        # int(mag[mag[exp['focused_feature']] == 10]['direction'])
        # this shows dist of all errors
        error_magnitude = df.groupby([exp['focused_feature'], 'error']).count().reset_index()
        # normalise to percentages using each experiment's individual total
        mag = error_magnitude.groupby(exp['focused_feature']).sum().reset_index()
        for p in mag[exp['focused_feature']]:
            # error_magnitude[error_magnitude[exp['focused_feature']] == p]['direction'] = \
            new_vals = error_magnitude[error_magnitude[exp['focused_feature']] == p]['direction'].div( float(mag[mag[exp['focused_feature']] == p]['direction']) ) * 100
            error_magnitude.loc[new_vals.index, 'direction'] = new_vals
        fig_error_magnitude = px.line(error_magnitude, x='error', y='direction', color=exp['focused_feature'])
        fig_error_magnitude.update_traces(mode='lines+markers')
        if show_titles:
            fig_error_magnitude.update_layout(title="Effect of %s on Error Magnitude distribution" % exp['focused_feature'])
        fig_error_magnitude.update_layout({'xaxis': {'title': {'text': 'Directional Error Magnitude'}}})
        fig_error_magnitude.update_layout({'yaxis': {'title': {'text': 'Percentage'}}})
        if len(df.groupby(exp['focused_feature'])) == 1: #hide legend if only one experiment
            fig_error_magnitude.update_layout(showlegend=False, coloraxis_showscale=False)
        if mag_legend_title_text:
            fig_error_magnitude.update_layout(legend_title_text=mag_legend_title_text)
        fig_error_magnitude.update_yaxes(type="log")


        # false positive is when classified directions is not actually that direction
        error_count = df[df['error'] != 0].groupby([exp['focused_feature'], 'chosen_direction']).count()
        total_count = df.groupby([exp['focused_feature'], 'chosen_direction']).count()
        error_percentage_FP = (error_count['error'].div(total_count['error'])*100).to_frame().reset_index()

        # this actually makes false negatives
        # error count
        error_count = df[df['error'] != 0].groupby([exp['focused_feature'], 'direction']).count()
        total_count = df.groupby([exp['focused_feature'], 'direction']).count()
        error_percentage_FN = (error_count['error'].div(total_count['error'])*100).to_frame().reset_index()


        # fig=px.bar(error_percentage_FN, x="direction", y="error", color=exp['focused_feature'])
        fig = go.Figure()
        FP = [k for k in error_percentage_FP.groupby(exp['focused_feature'])]
        FN = [k for k in error_percentage_FN.groupby(exp['focused_feature'])]
        #limit traces shown if neccesary for plotting in thesis to give clearer visibility
        # add FN
        for ex in FN:
            if limit_directional_error:
                if ex[0] in limit_directional_error:
                    fig.add_trace(go.Scatter(
                        name = "FN %d" % ex[0],
                        x = ex[1]['direction'],
                        y = ex[1]['error'],
                        mode='lines+markers'
                    ))
            else:
                fig.add_trace(go.Scatter(
                        name = "FN %d" % ex[0],
                        x = ex[1]['direction'],
                        y = ex[1]['error'],
                        mode='lines+markers'
                    ))
        # add FP
        for ex in FP:
            if limit_directional_error:
                if ex[0] in limit_directional_error:
                    fig.add_trace(go.Scatter(
                        name = "FP %d" % ex[0],
                        x = ex[1]['chosen_direction'],
                        y = ex[1]['error'],
                        mode='lines+markers'
                    ))
            else:
                fig.add_trace(go.Scatter(
                        name = "FP %d" % ex[0],
                        x = ex[1]['chosen_direction'],
                        y = ex[1]['error'],
                        mode='lines+markers'
                    ))
        if dir_legend_title_text:
            fig.update_layout(legend_title_text=dir_legend_title_text)
        if show_titles:
            fig.update_layout(title="Effect of %s on Directional Error" % exp['focused_feature'])
        fig.update_layout({'xaxis': {'title': {'text': 'Direction'}}})
        fig.update_layout({'yaxis': {'title': {'text': 'Percentage'}}})
        return fig, fig_error_magnitude

    def generateObjectRepresentationsL2(self, show_title=True):
        """Generate L2 object representation for each experiment

        Returns:
            (list): list containing object representations for each 
                    column. [experiments x [object representation x column]]
        """
        figs = []
        for exp in self.experiments:
            numCells = exp['params']['l2_params']['cellCount']
            total_objects = exp['params']['experiment']['iterations']
            cols = exp['params']['experiment']['num_cortical_columns']
            for iobj, obj in enumerate(exp['results']):
                ff = []
                for icol in range(cols):
                    im = np.empty((numCells, 0))
                    for iobj in range(total_objects):
                        temp = np.zeros((numCells, 1))
                        temp[list(obj['learnedObjects'][str(iobj)][icol])] = 1
                        im = np.hstack((im, temp))
                    fig = go.Figure()
                    fig.add_heatmap(z=im,
                                    xgap=1,
                                    colorscale=[[0, 'white'], [1.0, 'black']],
                                    showscale=False
                                    )
                    if show_title:
                        fig.update_layout(title='L2 Object Representations in C' + str(icol))
                    fig.update_layout(# height=numCells,
                                        yaxis_title="Activated Cells",
                                        xaxis_title="Object #",
                                        # width=400,
                                        height=800) 
                    ff.append(fig)
            figs.append(ff)
        return figs
    
    def generateCellActivityL2(self, show_title=True):
        """Generate L2 cell activations for each experiment.
        Figures are generated for every column of every object in every experiment
        [experiment x [objects x [cell activations x column]]]

        Returns:
            (list): list containing cell activations for each 
                    column for each object and experiment
        """
        figs = []
        match = 'Full L2 SDR'
        for iexp, exp in enumerate(self.experiments):
            object_figs = []
            for iobj, obj in enumerate(exp['results']):
                ff = []
                keys = [key for key in obj if match in key]
                numCells = exp['params']['l2_params']['cellCount']
                im = np.empty((numCells, 0))
                for key in keys:
                    for sensation in obj[key]:
                        temp = np.zeros((numCells, 1))
                        temp[sensation] = 1
                        im = np.hstack((im, temp))
                    
                    inferred_step = obj['touches']
                    vis = True
                    if inferred_step == None:
                        vis = False
                        inferred_step = 0

                    fig = go.Figure()
                    fig.add_heatmap(z=im,
                                    x=np.array(range(len(obj[key])))+1,#starts from 1, thus needs to be offset
                                    xgap=1,
                                    colorscale=[[0, 'white'], [1.0, 'black']],
                                    showscale=False
                                    ).add_shape(
                                                visible=vis,
                                                type='rect',
                                                x0=inferred_step - 0.5, x1=inferred_step + 1 - 0.5, y0=0 - 0.5, y1=numCells - 0.5,
                                                line_color='red'
                                            )
                    if obj['object'].get('augmentation', None) is not None:
                        aug = ' ' + obj['object']['augmentation']['type'] + '@' + str(obj['object']['augmentation']['amount'])
                    else:
                        aug = ''
                    if show_title:
                        fig.update_layout(title='Experiment #' + str(iexp) + ' | ' + key + ' Cell Activity for Object ' + str(iobj) + '(' + obj['object']['name'] + aug + ')')
                    fig.update_layout(# height=numCells,
                                        height=800,
                                        yaxis_title="Activated Cells",
                                        xaxis_title="Sensations") 
                    ff.append(fig)
                object_figs.append(ff)
            figs.append(object_figs)
        return figs

    def generateObjectRepresentationsL6(self, objects = [0,1,2], modules=[0,1,2], show_title=True, title_text=None, row_titles=None, rotation_col_titles=False):
        """Generate cell activations for all experiments

        Args:
            objects (list, optional): List of objects to include in subplots. Defaults to [0,1,2].
            modules (list, optional): List of modules to include in subplots. Defaults to [0,1,2].

        Returns:
            figs (list): List of figures for every experiment. Every figure shows the same modules and objects for comparison
        """
        # make it the same as previous one, with subfigures etc
        figs = []
        match = 'Full L6 SDR'
        df = pd.DataFrame(columns=['experiment', 'object', 'sensation', 'grid_cell_module', 'active_cells'])
        for iexp, exp in enumerate(self.experiments):
            numCells = exp['params']['experiment']['cells_per_axis']**2
            numModules =exp['params']['experiment']['num_modules']
            for obj in exp['results']:
                if int(obj['object']['name']) in objects:
                    keys = [key for key in obj if match in key]
                    for key in keys:
                        for isensation, sensation in enumerate(obj[key]):
                            temp = np.zeros((numCells*numModules, 1))
                            temp[sensation] = 1
                            module_activities = np.split(temp, numModules)
                            for m in modules:
                                row = {
                                    'experiment': iexp,
                                    'object' : int(obj['object']['name']),
                                    'sensation' : isensation,
                                    'grid_cell_module' : m,
                                    'active_cells' : module_activities[m].nonzero()[0].tolist()
                                }
                                df = df.append(pd.DataFrame(row, columns=['experiment', 'object', 'sensation', 'grid_cell_module', 'active_cells']), ignore_index=True)

        # this could also work in getting the active cells
        # df[(df['experiment']==0) & (df['object']=='0') & (df['sensation']==0) & (df['grid_cell_module']==0)]['active_cells']

        #obj cols and mods rows
        column_titles = ['Object ' + str(i) for i in objects]
        if not row_titles:
            row_titles = ['Module ' + str(i) for i in modules]
        
        for iexp, exp in enumerate(self.experiments):
            numCells = exp['params']['experiment']['cells_per_axis']**2
            if rotation_col_titles:
                column_titles = ['Object %s (rotated by %d degrees)' % (obj['object']['name'], obj['object']['augmentation']['amount']) for obj in exp['results'] if int(obj['object']['name']) in objects]

            fig = make_subplots(cols=len(objects),
                rows=len(modules),
                column_titles=column_titles,
                row_titles=row_titles,
                y_title="Cell Activations",
                x_title="Sensation") 

            for count_o, o in enumerate(objects):
                for count_m, m in enumerate(modules):
                    img = np.empty([numCells,0])
                    for i in range(exp['params']['experiment']['num_sensations']):
                        mask = df.loc[(df['experiment']==iexp) & (df['sensation'] == i) & (df['object'] == o) & (df['grid_cell_module'] == m), ['active_cells']].to_numpy()
                        #create binary sdr from active cells
                        col = np.zeros(numCells)
                        if len(np.array(mask[:,0].tolist())):
                            col[np.array(mask[:,0].tolist())] = 1
                        col = col.reshape((numCells,1))
                        img = np.concatenate((img, col), axis=1)

                    #starts from 1?
                    touches = [obj['touches'] for obj in exp['results'] if int(obj['object']['name']) == o]
                    assert len(touches) == 1, 'The len of touches should only be one (this is %d)' % len(touches)
                    inferred_step = touches[0]
                    # print('Experiment %d Object %d module %d inferred step %d' % (iexp, o,m, inferred_step))

                    vis = True
                    if inferred_step == None:
                        vis = False
                        inferred_step = 0
                    # else:
                    
                    #     inferred_step = inferred_step - 1 #I think it starts from 1 not 0
                    fig.add_heatmap(z=img,
                                    x=np.array(range(exp['params']['experiment']['num_sensations']))+1,#starts from 1, thus needs to be offset
                                    colorscale=[[0, 'white'], [1.0, 'black']],
                                    showscale=False,
                                    row=count_m+1,
                                    col=count_o+1#starts from 1
                                    ).add_shape(
                                        visible=vis,
                                        type='rect',
                                        x0=inferred_step - 0.5, x1=inferred_step + 1 - 0.5, y0=0 - 0.5, y1=numCells - 0.5,
                                        line_color='red',
                                        row=count_m+1,
                                        col=count_o+1
                                    )
            if show_title and title_text:
                fig.update_layout(title=title_text)
            elif show_title:
                fig.update_layout(title='Experiment #' + str(iexp) + ' (' + str(exp['focused_feature']) + ' = ' + str(exp['params']['experiment'][exp['focused_feature']]) + ')')
            fig.update_layout(height=800)
            figs.append(fig)
        return figs

    def combine_errors(self, fig1, fig2, show_title_text=True, title_text=None):
        # plot errors of 2 experiments togther. fig one is at the bottom
        data_y_1 = fig1['data'][0]['y']
        data_x_1 = fig1['data'][0]['x']
        x_axis_title_text_1 = fig1['layout']['xaxis']['title']['text']

        data_y_2 = fig2['data'][0]['y']
        data_x_2 = fig2['data'][0]['x']
        x_axis_title_text_2 = fig2['layout']['xaxis']['title']['text']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data_x_1,
            y=data_y_1,
            name=x_axis_title_text_1
        ))
        fig.add_trace(go.Scatter(
            x=data_x_2,
            y=data_y_2,
            name=x_axis_title_text_2,
            xaxis="x2"
        ))

        if show_title_text and title_text:
            fig.update_layout(title_text=title_text)
        elif show_title_text:
            fig.update_layout(title_text="Comparison between %s and %s on inference error" % (x_axis_title_text_1, x_axis_title_text_2))


        # Create axis objects
        fig.update_layout(
            {
                'yaxis': {'title': 'Error'},
                'xaxis': {
                    'title': x_axis_title_text_1,
                },
                'xaxis2': {
                    'title': x_axis_title_text_2,
                    # 'titlefont': {'color': '#1f77b4'},
                    # 'tickfont': {'color': '#1f77b4'},
                    'overlaying': 'x',
                    'side': 'top'
                }
            }
        )

        return fig

    # def generateCorrectClassificationHist(self, title_text=None, legend_title_text=None, show_title=True, y_axis_title=None):
    #     #basically generate a cumulative accuracy graph?

    def generateOrientationDist(self, direction, show_title=True, title_text=None, exp_num=0): # hardcoded to only show first exp
        # makes you see false positives
        # make a distribution for a direction's selectivity
        # basically combine all objects's real orientations that were classified by a certain chosen direction (orientation)
        fig = go.Figure()

        # get hist of chosen directions
        data = {
            'direction':[],
            self.experiments[0]['focused_feature']: []
        }
        for exp in self.experiments:
            for obj in exp['results']:
                if obj['chosen_orientation'] == direction:
                    data['direction'].append(obj['object']['augmentation']['amount'])
                    data[exp['focused_feature']].append(exp['params']['experiment'][exp['focused_feature']])
        df = pd.DataFrame(data, columns=[exp['focused_feature'], 'direction'])
        group_labels = [str(ex[0]) for ex in df.groupby(exp['focused_feature'])]
        hist_data = [df[df[exp['focused_feature']] == int(group_label)]['direction'] for group_label in group_labels]

        # fig = figfact.create_distplot(hist_data, group_labels)
        # fig = px.histogram(df, x='direction', color=exp['focused_feature'],
        #                      nbins=exp['params']['experiment']['num_modules'],
        #                      range_x=[0,360],
        #                      histnorm='probability density',
        #                      barmode='overlay')
        fig.add_histogram(x=hist_data[exp_num], 
                            xbins={ #shift half a bar left, because receptive fields (module orientations) is actually centered around rotation poins
                                'start': str(0 - (360/self.experiments[exp_num]['params']['experiment']['num_modules'])/2),
                                'end': str(360 - (360/self.experiments[exp_num]['params']['experiment']['num_modules'])/2),
                                'size': str(360/self.experiments[exp_num]['params']['experiment']['num_modules'])
                            },
                            histnorm='percent',
                            name='Orientation Selectivity'
                            # nbinsx=exp['params']['experiment']['num_modules'], 
                            # autobinx=False
                            )
        
        # get firing rates and make scatter plot of their sum
        # modules x cols x sensations
        data = {
            'firing_rate':[],
            'direction':[],
            self.experiments[0]['focused_feature']: []
        }
        for exp in self.experiments:
            perModRange=exp['params']['experiment']['angle']/exp['params']['experiment']['num_modules']
            for obj in exp['results']:
                if obj['chosen_orientation'] == direction: # must be same conditional as hist
                    for iorientation, orientation in enumerate(obj['orientational_firing_rate']):
                        for col in orientation:
                            for isensation, rate in enumerate(col):
                                # first make for individual cols then agrigated?
                                data['direction'].append(iorientation*perModRange)
                                if rate==0:
                                    data['firing_rate'].append(1/1000)
                                    print('somethin fishy here')
                                else:
                                    data['firing_rate'].append(rate)
                                # data['sensation'].append(isensation+1)
                                data[exp['focused_feature']].append(exp['params']['experiment'][exp['focused_feature']])
        df = pd.DataFrame(data, columns=[exp['focused_feature'], 'direction', 'firing_rate'])
        df = df.groupby(['direction', exp['focused_feature']], as_index=False).sum()
        
        fig.add_scatter(x=np.array(df[df[exp['focused_feature']] == int(group_labels[exp_num])]['direction']) ,#+ (360/self.experiments[exp_num]['params']['experiment']['num_modules'])/2, #shift to the middle of bars
                        y=np.array(df[df[exp['focused_feature']] == int(group_labels[exp_num])]['firing_rate']) / max(df[df[exp['focused_feature']] == int(group_labels[exp_num])]['firing_rate']), # normal;ise
                        yaxis="y2",
                        name='Firing Rate')

        fig.update_layout(
            {
                'xaxis': {'title': 'Orientation Bins (°)'},
                'yaxis': {
                    'title': 'Percent',
                },
                'yaxis2': {
                    'title': 'Firing Rate',
                    # 'titlefont': {'color': '#1f77b4'},
                    # 'tickfont': {'color': '#1f77b4'},
                    'overlaying': 'y',
                    'side': 'right'
                }
            }
        )
        fig.update_xaxes(range=[(0 - (360/self.experiments[exp_num]['params']['experiment']['num_modules'])/2),(360 - (360/self.experiments[exp_num]['params']['experiment']['num_modules'])/2)])
        if show_title and title_text:
                fig.update_layout(title=title_text)
        elif show_title:
            fig.update_layout(title='Experiment %d (%s=%d) | Orientational Selectivity in Direction %d' % 
            (exp_num, 
            exp['focused_feature'], 
            self.experiments[exp_num]['params']['experiment'][exp['focused_feature']], 
            direction))
        return fig

    def generateOrientationDistFN(self, direction, show_title=True, title_text=None, exp_num=0): # hardcoded to only show first exp
        # makes you see false positives
        # make a distribution for a direction's selectivity
        # basically combine all objects's real orientations that were classified by a certain chosen direction (orientation)
        fig = go.Figure()

        # get hist of chosen directions
        data = {
            'direction':[],
            self.experiments[0]['focused_feature']: []
        }
        for exp in self.experiments:
            perModRange=exp['params']['experiment']['angle']/exp['params']['experiment']['num_modules']
            for obj in exp['results']:
                if obj['ideal_orientation'] == direction:
                    data['direction'].append(obj['chosen_orientation']*perModRange)
                    data[exp['focused_feature']].append(exp['params']['experiment'][exp['focused_feature']])
        df = pd.DataFrame(data, columns=[exp['focused_feature'], 'direction'])
        group_labels = [str(ex[0]) for ex in df.groupby(exp['focused_feature'])]
        hist_data = [df[df[exp['focused_feature']] == int(group_label)]['direction'] for group_label in group_labels]

        # fig = figfact.create_distplot(hist_data, group_labels)
        # fig = px.histogram(df, x='direction', color=exp['focused_feature'],
        #                      nbins=exp['params']['experiment']['num_modules'],
        #                      range_x=[0,360],
        #                      histnorm='probability density',
        #                      barmode='overlay')
        fig.add_histogram(x=hist_data[exp_num], 
                            xbins={ #shift half a bar left, because receptive fields (module orientations) is actually centered around rotation poins
                                'start': str(0 - (360/self.experiments[exp_num]['params']['experiment']['num_modules'])/2),
                                'end': str(360 - (360/self.experiments[exp_num]['params']['experiment']['num_modules'])/2),
                                'size': str(360/self.experiments[exp_num]['params']['experiment']['num_modules'])
                            },
                            histnorm='percent',
                            name='Orientation Selectivity'
                            # nbinsx=exp['params']['experiment']['num_modules'], 
                            # autobinx=False
                            )
        
        # get firing rates and make scatter plot of their sum
        # modules x cols x sensations
        data = {
            'firing_rate':[],
            'direction':[],
            self.experiments[0]['focused_feature']: []
        }
        for exp in self.experiments:
            perModRange=exp['params']['experiment']['angle']/exp['params']['experiment']['num_modules']
            for obj in exp['results']:
                if obj['ideal_orientation'] == direction: # must be same conditional as hist
                    for iorientation, orientation in enumerate(obj['orientational_firing_rate']):
                        for col in orientation:
                            for isensation, rate in enumerate(col):
                                # first make for individual cols then agrigated?
                                data['direction'].append(iorientation*perModRange)
                                if rate==0:
                                    data['firing_rate'].append(1/1000)
                                    print('somethin fishy here')
                                else:
                                    data['firing_rate'].append(rate)
                                # data['sensation'].append(isensation+1)
                                data[exp['focused_feature']].append(exp['params']['experiment'][exp['focused_feature']])
        df = pd.DataFrame(data, columns=[exp['focused_feature'], 'direction', 'firing_rate'])
        df = df.groupby(['direction', exp['focused_feature']], as_index=False).sum()
        
        fig.add_scatter(x=np.array(df[df[exp['focused_feature']] == int(group_labels[exp_num])]['direction']) ,#+ (360/self.experiments[exp_num]['params']['experiment']['num_modules'])/2, #shift to the middle of bars
                        y=np.array(df[df[exp['focused_feature']] == int(group_labels[exp_num])]['firing_rate']) / max(df[df[exp['focused_feature']] == int(group_labels[exp_num])]['firing_rate']), # normal;ise
                        yaxis="y2",
                        name='Firing Rate')

        fig.update_layout(
            {
                'xaxis': {'title': 'Direction'},
                'yaxis': {
                    'title': 'Percent',
                },
                'yaxis2': {
                    'title': 'Normalised Firing Rate',
                    # 'titlefont': {'color': '#1f77b4'},
                    # 'tickfont': {'color': '#1f77b4'},
                    'overlaying': 'y',
                    'side': 'right'
                }
            }
        )
        fig.update_xaxes(range=[(0 - (360/self.experiments[exp_num]['params']['experiment']['num_modules'])/2),(360 - (360/self.experiments[exp_num]['params']['experiment']['num_modules'])/2)])
        if show_title and title_text:
                fig.update_layout(title=title_text)
        elif show_title:
            fig.update_layout(title='Experiment %d (%s=%d) | Orientational Selectivity FN in Direction %d' % 
            (exp_num, 
            exp['focused_feature'], 
            self.experiments[exp_num]['params']['experiment'][exp['focused_feature']], 
            direction))
        return fig

        # maak n graph wat die orientational selectivity anders uitwerk?? kyk (per orientation) (met die moontlikheid vir die ideal case) wat die firing rates vir alle rigrings is en maak n histogram??
        # ek dink n gaussian dis is makliker om te lees as n sirkel storie (maar sirkel lyk ook nice om in te gooir tho)

    def compareOrientationAlgoAccuracy(self):
        # basically get each touches and make a cumalitive graph over sensations
        data = {
            'touches': [],
            self.experiments[0]['focused_feature']: []
        }
        for exp in self.experiments:
            for obj in exp['results']:
                if obj['touches'] == None:
                    data['touches'].append(0)
                else:
                    data['touches'].append(obj['touches'])
                data[exp['focused_feature']].append(exp['params']['experiment'][exp['focused_feature']])
        df = pd.DataFrame(data, columns=[exp['focused_feature'], 'touches'])
        data = {
            'accuracy': [],
            'sensation': [],
            exp['focused_feature']: []
        }
        cumulative = {} #keep a running total
        for a in df.groupby(exp['focused_feature']).count().reset_index()['orientationAlgo']:
            cumulative[str(a)] = 0
        for idx, d in df.groupby([exp['focused_feature'],'touches']):
            algo = idx[0]
            sensation = idx[1]
            count = d.count()['touches']
            tmp = df.groupby(exp['focused_feature']).count().reset_index()
            accuracy = count/int(tmp[tmp[exp['focused_feature']] == algo]['touches'])
            if sensation != 0:
                cumulative[str(algo)] += accuracy
            data['sensation'].append(sensation)
            data['accuracy'].append(cumulative[str(algo)])
            data[exp['focused_feature']].append(algo)
        plot_data = df = pd.DataFrame(data, columns=[exp['focused_feature'], 'accuracy', 'sensation'])
        fig = go.Figure()
        tracenames = ['Base', 'Ideal', 'Orientation']
        for trace in plot_data.groupby(exp['focused_feature']).count().reset_index()['orientationAlgo']:
            fig.add_trace(
                go.Scatter(
                    x=plot_data[plot_data[exp['focused_feature']]==trace]['sensation'],
                    y=plot_data[plot_data[exp['focused_feature']]==trace]['accuracy'],
                    name=tracenames[trace]
                )
            )
        fig.update_layout({'xaxis': {'title': {'text': 'Sensation'}}})
        fig.update_layout({'yaxis': {'title': {'text': 'Cumulative Accuracy'}}})
        fig.update_layout(legend_title_text='Algorithm')
        return fig

    def generateSensationsVsColumnsOverObjects(self, legend_title_text=None, show_avg_sensations=False):
        # similar to 2017 paper results
        data = {
            'iterations': [],
            'num_cortical_columns': [],
            'avg_sensations': [],
            'accuracy': []
        }
        nn = 0
        for exp in self.experiments:
            iterations = exp['params']['experiment']['iterations']
            num_cortical_columns = exp['params']['experiment']['num_cortical_columns']
            avg_sensations = 0.0 #only include successful ones??
            accuracy = 0.0
            for obj in exp['results']: #maybe use median? maybe use confidence interval?
                if obj['touches'] == None:
                    avg_sensations+=10
                    nn+=1
                else:
                    avg_sensations+=obj['touches']
                    accuracy+=1.0
            avg_sensations/=len(exp['results'])
            accuracy/=len(exp['results'])
            data['iterations'].append(iterations)
            data['num_cortical_columns'].append(num_cortical_columns)
            data['avg_sensations'].append(avg_sensations)
            data['accuracy'].append(accuracy)
        df = pd.DataFrame(data, columns=['iterations', 'num_cortical_columns', 'avg_sensations', 'accuracy'])
        if show_avg_sensations:
            fig = px.line(df, x="iterations", y="avg_sensations", color='num_cortical_columns')
            fig.update_layout({'yaxis': {'title': {'text': 'Average Number of Sensations'}}})
        else:
            fig = px.line(df, x="iterations", y="accuracy", color='num_cortical_columns')
            fig.update_layout({'yaxis': {'title': {'text': 'Accuracy'}}})

        fig.update_layout({'xaxis': {'title': {'text': 'Number of Learned Objects'}}})
        fig.update_layout(legend_title_text='Number of Columns')
        print(nn)
        return fig

    def generateSensationsVsColumnsOverUniqueFeatures(self):
        # similar to 2017 paper results
        data = {
            'num_features': [],
            'num_cortical_columns': [],
            'avg_sensations': [],
            'accuracy': []
        }
        nn = 0
        for exp in self.experiments:
            num_features = exp['params']['experiment']['num_features']
            num_cortical_columns = exp['params']['experiment']['num_cortical_columns']
            avg_sensations = 0.0 #only include successful ones??
            accuracy = 0.0
            for obj in exp['results']: #maybe use median? maybe use confidence interval?
                if obj['touches'] == None:
                    avg_sensations+=10
                    nn+=1
                else:
                    avg_sensations+=obj['touches']
                    accuracy+=1.0
            avg_sensations/=len(exp['results'])
            accuracy/=len(exp['results'])
            data['num_features'].append(num_features)
            data['num_cortical_columns'].append(num_cortical_columns)
            data['avg_sensations'].append(avg_sensations)
            data['accuracy'].append(accuracy)
        df = pd.DataFrame(data, columns=['num_features', 'num_cortical_columns', 'avg_sensations', 'accuracy'])
        fig = px.line(df, x="num_features", y="accuracy", color='num_cortical_columns')
        fig.update_layout({'xaxis': {'title': {'text': 'Number of Unique Features'}}})
        fig.update_layout({'yaxis': {'title': {'text': 'Accuracy'}}})
        fig.update_layout(legend_title_text='Number of Columns')
        fig.update_xaxes(autorange="reversed")
        print(nn)
        return fig
            
    def generateMultiObject(self):
        #comparison on performance of all object dimentiona and sensations
        data = {
            'iterations': [],
            'num_features': [],
            'features_per_object': [],
            'avg_sensations': [],
            'accuracy': []
        }
        for exp in self.experiments:
            iterations = exp['params']['experiment']['iterations']
            num_features = exp['params']['experiment']['num_features']
            features_per_object = exp['params']['experiment']['features_per_object']
            avg_sensations = 0.0 
            accuracy = 0.0
            for obj in exp['results']:
                if obj['touches'] == None:
                    avg_sensations+=10
                else:
                    avg_sensations+=obj['touches']
                    accuracy+=1.0
            avg_sensations/=len(exp['results'])
            accuracy/=len(exp['results'])
            data['iterations'].append(iterations)
            data['num_features'].append(num_features)
            data['features_per_object'].append(features_per_object)
            data['avg_sensations'].append(avg_sensations)
            data['accuracy'].append(accuracy)
        df = pd.DataFrame(data, columns=['iterations', 'num_features', 'features_per_object', 'avg_sensations', 'accuracy'])
        fig = go.Figure()
        for i, nf in enumerate(df.groupby('num_features')):
            num_feat = nf[0]
            c = colors[i]
            for j, fpo in enumerate(nf[1].groupby('features_per_object')):
                feat_obj = fpo[0]
                fig.add_trace(
                    go.Scatter(
                        x=fpo[1]['iterations'],
                        y=fpo[1]['accuracy'],
                        line=dict(color=c, dash=line_dash_shape[j]),
                        name='%d %d' % (feat_obj, num_feat)
                    )
                )
        fig.update_layout({'xaxis': {'title': {'text': 'Number of Learned Objects'}}})
        fig.update_layout({'yaxis': {'title': {'text': 'Accuracy'}}})
        fig.update_layout(legend_title_text='(Features per Object, Unique Pool Size)')
        return fig

    def generateMultiObjectCellsNumber(self):
        # mutiple cells per axis, number of modules, and iterations
        data = {
            'iterations': [],
            'num_modules': [],
            'cells_per_axis': [],
            'avg_sensations': [],
            'accuracy': []
        }        
        for exp in self.experiments:
            iterations = exp['params']['experiment']['iterations']
            num_modules = exp['params']['experiment']['num_modules']
            cells_per_axis = exp['params']['experiment']['cells_per_axis']
            avg_sensations = 0.0 
            accuracy = 0.0
            for obj in exp['results']:
                if obj['touches'] == None:
                    avg_sensations+=10
                else:
                    avg_sensations+=obj['touches']
                    accuracy+=1.0
            avg_sensations/=len(exp['results'])
            accuracy/=len(exp['results'])
            data['iterations'].append(iterations)
            data['num_modules'].append(num_modules)
            data['cells_per_axis'].append(cells_per_axis)
            data['avg_sensations'].append(avg_sensations)
            data['accuracy'].append(accuracy)
        df = pd.DataFrame(data, columns=['iterations', 'num_modules', 'cells_per_axis', 'avg_sensations', 'accuracy'])
        fig = go.Figure()
        for i, nf in enumerate(df.groupby('cells_per_axis')):
            num_feat = nf[0]
            c = colors[i]
            for j, fpo in enumerate(nf[1].groupby('num_modules')):
                feat_obj = fpo[0]
                fig.add_trace(
                    go.Scatter(
                        x=fpo[1]['iterations'],
                        y=fpo[1]['accuracy'],
                        line=dict(color=c, dash=line_dash_shape[j]),
                        name='%d %d' % (feat_obj, num_feat)
                    )
                )
        fig.update_layout({'xaxis': {'title': {'text': 'Number of Learned Objects'}}})
        fig.update_layout({'yaxis': {'title': {'text': 'Accuracy'}}})
        fig.update_layout(legend_title_text='(Number of Modules, Cells per Axis)')
        return fig

    def generateMultiObjectUniqueFeatCellsNumber(self, show_avg_sensations=False):
        # mutiple cells per axis, number of modules, and iterations
        data = {
            'num_features': [],
            'num_modules': [],
            'cells_per_axis': [],
            'avg_sensations': [],
            'accuracy': []
        }        
        for exp in self.experiments:
            num_features = exp['params']['experiment']['num_features']
            num_modules = exp['params']['experiment']['num_modules']
            cells_per_axis = exp['params']['experiment']['cells_per_axis']
            avg_sensations = 0.0 
            accuracy = 0.0
            for obj in exp['results']:
                if obj['touches'] == None:
                    avg_sensations+=10
                else:
                    avg_sensations+=obj['touches']
                    accuracy+=1.0
            avg_sensations/=len(exp['results'])
            accuracy/=len(exp['results'])
            data['num_features'].append(num_features)
            data['num_modules'].append(num_modules)
            data['cells_per_axis'].append(cells_per_axis)
            data['avg_sensations'].append(avg_sensations)
            data['accuracy'].append(accuracy)
        df = pd.DataFrame(data, columns=['num_features', 'num_modules', 'cells_per_axis', 'avg_sensations', 'accuracy'])
        fig = go.Figure()
        for i, nf in enumerate(df.groupby('cells_per_axis')):
            num_feat = nf[0]
            c = colors[i]
            for j, fpo in enumerate(nf[1].groupby('num_modules')):
                feat_obj = fpo[0]
                if show_avg_sensations:
                    y = fpo[1]['avg_sensations']
                else:
                    y = fpo[1]['accuracy']
                fig.add_trace(
                    go.Scatter(
                        x=fpo[1]['num_features'],
                        y=y,
                        line=dict(color=c, dash=line_dash_shape[j]),
                        name='%d %d' % (feat_obj, num_feat)
                    )
                )
        if show_avg_sensations:
            fig.update_layout({'yaxis': {'title': {'text': 'Average number of Sensations'}}})
        else:
            fig.update_layout({'yaxis': {'title': {'text': 'Accuracy'}}})
        fig.update_layout({'xaxis': {'title': {'text': 'Number of Unique Features'}}})
        fig.update_layout(legend_title_text='(Number of Modules, Cells per Axis)')
        fig.update_xaxes(autorange="reversed")
        return fig


    def generate_t_results(self, file_list):
        # this is made for orientationalgo comparison so far
        data = {}

        for i_iter, iteration in enumerate(file_list):
            with open(os.path.join(SCRIPT_DIR, iteration)) as f:
                experiments = json.load(f)
            
            #first init
            if i_iter == 0:
                data['touches'] = []
                data[experiments[0]['focused_feature']] = []
                data['iteration'] = []

            # data = {
            #     'touches': [],
            #     experiments[0]['focused_feature']: [],
            #     'iteration': []
            # }
            for exp in experiments:
                for obj in exp['results']:
                    if obj['touches'] == None:
                        data['touches'].append(0)
                    else:
                        data['touches'].append(obj['touches'])
                    data[exp['focused_feature']].append(exp['params']['experiment'][exp['focused_feature']])
                    data['iteration'].append(i_iter)
        
        df = pd.DataFrame(data, columns=[exp['focused_feature'], 'touches', 'iteration'])

        totals = df.groupby(['iteration', exp['focused_feature']]).count()
        classified = df[df['touches'] != 0].groupby(['iteration', exp['focused_feature']]).count()
        accuracy = classified/totals
        accuracy_sampled = defaultdict(list)
        # make all the accuracy samples
        for idx, d in accuracy.groupby('orientationAlgo'):
            accuracy_sampled[idx].extend(d.values.flatten())

        print('The accuracy sampled is:')
        print(accuracy_sampled)

    def compareMultipleOrientationAlgoAccuracy(self, file_list):
        fig = go.Figure()
        data_base = []
        data_ideal = []
        data_orientation = []
        percentiles = [5, 50, 95]
        for i_iter, iteration in enumerate(file_list):
            with open(os.path.join(SCRIPT_DIR, iteration)) as f:
                self.experiments = json.load(f)
                plot = self.compareOrientationAlgoAccuracy()
                data_base.append(plot._data[0]['y'])
                data_ideal.append(plot._data[1]['y'])
                data_orientation.append(plot._data[2]['y'])
                print('lel')
        p1,p2,p3 = np.percentile(np.array(data_base), percentiles, axis=0)
        data = p2.tolist()
        error_below = (p2-p1).tolist()
        error_above = (p3-p2).tolist()
        #jsut add them manually for now
        fig.add_trace(
                go.Scatter(
                    x=plot._data[0]['x'].tolist(),
                    y=data,
                    name = 'Base',
                    mode = "lines+markers",
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=error_above,
                        arrayminus=error_below
                    )
                )
            )
        p1,p2,p3 = np.percentile(np.array(data_ideal), percentiles, axis=0)
        data = p2.tolist()
        error_below = (p2-p1).tolist()
        error_above = (p3-p2).tolist()
        #jsut add them manually for now
        fig.add_trace(
                go.Scatter(
                    x=plot._data[0]['x'].tolist(),
                    y=data,
                    name = 'Ideal',
                    mode = "lines+markers",
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=error_above,
                        arrayminus=error_below
                    )
                )
            )
        p1,p2,p3 = np.percentile(np.array(data_orientation), percentiles, axis=0)
        data = p2.tolist()
        error_below = (p2-p1).tolist()
        error_above = (p3-p2).tolist()
        #jsut add them manually for now
        fig.add_trace(
                go.Scatter(
                    x=plot._data[0]['x'].tolist(),
                    y=data,
                    name = 'Orientation',
                    mode = "lines+markers",
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=error_above,
                        arrayminus=error_below
                    )
                )
            )
        fig.update_layout({'xaxis': {'title': {'text': 'Sensation'}}})
        fig.update_layout({'yaxis': {'title': {'text': 'Cumulative Accuracy'}}})
        fig.update_layout(legend_title_text='Algorithm')
        return fig

    def generateMultiLine(self):
        print('dc')
        # >>>>>>>>location layer<<<<<<<<<<<<<
        # num_modules
        # cells_per_axis
        # >>>>>>>>object generation<<<<<<<<<<<<<
        # iterations
        # num_features
        # features_per_object
        # >>>>>>>>general netowork<<<<<<<<<<<<<
        # num_cortical_columns
        # num_sensations

import plotly.io as pio
pio.kaleido.scope.default_width = 750
pio.kaleido.scope.default_height = 480
pio.kaleido.scope.default_scale = 2

if __name__ == '__main__':
    with open(os.path.join(SCRIPT_DIR, "validation_results/25_13_dist_result.json")) as f:
        data = json.load(f)
    pp = GeneratePlots(data)
    savefile="validation_results/25_13_dist_result.json"
    pp.generateOrientationDist(18, show_title=False).update_layout(font_size=15).write_image("%s/%s.png" % (savefile.split('/')[0], 'orientation_18_selectivity_25_13'))
    pp.generateOrientationDist(10, show_title=False).update_layout(font_size=15).write_image("%s/%s.png" % (savefile.split('/')[0], 'orientation_10_selectivity_25_13'))
    # pp.generateMultiObject().show()
    # pp.compareOrientationAlgoAccuracy().show()
    # pp.generateSensationsVsColumnsOverUniqueFeatures().show()
    # pp.generateMultiObjectCellsNumber().show()
    # pp.generateMultiObjectUniqueFeatCellsNumber().show()
    # pp.generateSensationsVsColumnsOverObjects().show()
    # [[pp.generateOrientationDist(i, exp_num=j).show() for i in range(25)] for j in range(1)]
    # pp.generateOrientationDist(18).show()
    # pp.generateOrientationDistFN(18).show()
    # pp.generateOrientationDist(15).show()
    # pp.generateOrientationDist(16).show()
    # pp.generateOrientationDist(17).show()
    # with open(os.path.join(SCRIPT_DIR, "validation_results/num_modules_result.json")) as f:  cells_per_axis_result
    #     data = json.load(f)
    # pp2 = GeneratePlots(data)
    # pp.generateDensity('L6a Representation', normalization_key='Location')
    # pp.generateDensityRidge('L6a Representation').show()
    # pp.generateTouchesHist().show()
    # pp.generateInferredDist().show()
    # de_dist, de_mag = pp.generateDirectionalError(mag_legend_title_text='Number of Modules', dir_legend_title_text='(FP,FN) Number of Modules')
    # de_dist.show()
    # de_mag.show()
    # de_mag.show()
    # pp.generateDensity('L6a Representation',
    #                     yaxis_title='Cell Activation Density',
    #                     normalization_key='Location').show()
    # x = pp.generateObjectRepresentationsL2()
    # ll = pp.generateCellActivityL2()
    # ll[0][2][0].show()
    # ll[0][3][0].show()
    # ll[1][2][0].show()
    # ll[1][3][0].show()
    # ll[2][2][0].show()
    # ll[2][3][0].show()
    # ll[3][2][0].show()
    # ll[3][3][0].show()
    # pp.generateObjectRepresentationsL6(objects = [5], modules=[1,2,3], rotation_col_titles=False)[0].show()
    # cc = pp.generateObjectRepresentationsL6(objects = [1,2,3, 4,5,6], modules=[0,1,2,3,4,5,6], rotation_col_titles=True)
    # cc[0].show()
    # cc[1].show()
    # cc[2].show()
    # cc[3].show()
    # cc[4].show()
    # cc[5].show()
    # cc[6].show()
    # pp.combine_errors(pp2.generateInferredDist('Number of Modules'), pp.generateInferredDist('Cells per Axis')).show()
    # data_y = k['data'][0]['y']
    # data_x = k['data'][0]['y']
    # x_axis_title_text = k['layout']['xaxis']['title']['text']
    # pp.combine_errors()
    # pp.generateInferredDist().show()
    # pp.generateDirectionalError().show()
    # pp.generateDirectionalSelectivityStacked(0).show()
    # pp.generateDirectionalSelectivityStacked(1).show()
    # pp.generateDirectionalSelectivityStacked(2).show()
    # pp.generateDirectionalSelectivityStacked(3).show()
    # pp.generateDirectionalSelectivityStacked(4).show()
    # pp.generateDirectionalSelectivityStacked(5).show()
    # pp.generateDirectionalSelectivityStacked(6).show()
    # pp.generateDirectionalSelectivityStacked(7).show()
    # pp.generateDirectionalSelectivityStacked(8).show()
    # pp.generateDirectionalSelectivityStacked(9).show()
