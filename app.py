
import os
from h2o_wave import main, app, Q, ui, on, handle_on, data
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelEncoder
import statistics
import base64
import io

warnings.filterwarnings('ignore')


class MyLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


train_df = pd.read_csv(
    r'data\new_train_after_outlier_removal.csv')
orig_train_df = pd.read_csv(
    r'data\train.csv')
# values = train_df.values.tolist()
# variable_names = train_df.columns.tolist()

orig_train_df["ID"] = orig_train_df['StateCode'] + \
    "-" + orig_train_df["StationId"].astype(str)
mle = MyLabelEncoder()
mle.fit(orig_train_df['ID'])
orig_train_df['ID'] = mle.transform(orig_train_df['ID'])
orig_train_df['ID'] = orig_train_df['ID'].astype('int64')

# Variables for UI
Id = 0
col_selected = 'AQI'
is_outlier = True


rows_per_page = 10
total_rows = len(orig_train_df)


def df_to_table_rows(df: pd.DataFrame) -> List[ui.TableRow]:
    return [ui.table_row(name=str(r[0]), cells=[str(r[x]) for x in range(len(df.columns))]) for r in df.itertuples(index=False)]


def get_df(base: pd.DataFrame, sort: Dict[str, bool] = None, search: Dict = None, filters: Dict[str, List[str]] = None) -> pd.DataFrame:
    # Make a deep copy in order to not mutate the original df which serves as our baseline.
    df = base.copy()

    if sort:
        # Reverse values since default sort of Wave table is different from Pandas.
        ascending = [not v for v in list(sort.values())]
        df = df.sort_values(by=list(sort.keys()), ascending=ascending)
    # Filter out all rows that do not contain searched string in `text` cell.
    if search:
        search_val = search['value'].lower()
        # Filter dataframe by search value case insensitive.
        df = df[df.apply(lambda r: any(search_val in str(r[col]).lower()
                                       for col in search['cols']), axis=1)]
    # Filter out rows that do not contain filtered column value.
    if filters:
        # We want only rows that have no filters applied or their col value matches active filters.
        query = ' & '.join(
            [f'({not bool(filters)} | {col} in {filters})' for col, filters in filters.items()])
        df = df.query(query)

    return df


def make_col_rows(df, id, col):
    df = df[df['ID'] == id]
    if col == 'PM25':
        col = 'PM2.5'
    print('\x1b[6;30;42m'+col+'\x1b[0m')
    print(df[col])
    return list(zip(df['Date'], df[col]))


def current_outlier_selection(is_outlier):
    if is_outlier:
        return 'Outlier Detection'
    return 'Feature Visualization'


def ui_id_list(df, col):
    l = []
    for u in sorted(df[col].unique().tolist()):
        l.append(ui.choice(name=f'choice{u}', label=f'ID {u}'))
    return l


def ui_graph_type_choices():
    l = []
    l.append(ui.choice(name='Feature Visualization',
                       label='Feature Visualization'))
    l.append(ui.choice(name='Outlier Detection', label='Outlier Detection'))
    return l


def ui_cols_list(df):
    l = []
    for u in sorted(df.columns.tolist()):
        if u == 'PM2.5':
            l.append(ui.choice(name='PM25', label='PM25'))
        elif u in ['PM10', 'CO', 'SO2', 'O3', 'AQI']:
            l.append(ui.choice(name=f'{u}', label=f'{u}'))
    return l


def box_plot_list(df, id, col):
    df = df[df['ID'] == id]
    if col == 'PM25':
        col = 'PM2.5'
    # finding the Q1(25 percentile) and Q3(75 percentile)
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    q2 = statistics.median(df[col].tolist())

    # finding out the value of Inter Quartile Range
    IQR = q3 - q1

    # defining max and min limits
    high = q3 + (1.5 * IQR)
    low = q1 - (1.5 * IQR)

    return [[col, low, q1, q2, q3, high]]


def outlier_detection(df, feature, status):
    print('\x1b[6;30;42m'+feature+'\x1b[0m')
    df = df[df['ID'] == Id]
    if feature == 'PM25':
        feature = 'PM2.5'
    plt.figure(figsize=(4, 3))
    plt.suptitle(f"Distribution {status} handling for {feature}", fontsize=11)
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=df[feature])
    print(df[feature])
    plt.subplot(1, 2, 2)
    df = df.reset_index()
    sns.boxplot(data=df[feature], palette="magma")
    plt.tight_layout()
    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    image = base64.b64encode(pic_IObytes.read()).decode()
    return image


def make_table_columns(df):
    l = []
    for col in df.columns.tolist():
        l.append(ui.table_column(name=f'{col}', label=f'{col}', sortable=True))
    return l


def make_table_rows():
    l = []
    for index, row in orig_train_df.iterrows():
        print(row.tolist())
        x = ui.table_row(
            name=str(index),
            cells=[str(y) for y in row.tolist()]
        )
        l.append(x)
    return l

# Use for page cards that should be removed when navigating away.
# For pages that should be always present on screen use q.page[key] = ...


def add_card(q, name, card) -> None:
    q.client.cards.add(name)
    q.page[name] = card
    return q


# Remove all the cards related to navigation.
def clear_cards(q, ignore: Optional[List[str]] = []) -> None:
    if not q.client.cards:
        return

    for name in q.client.cards.copy():
        if name not in ignore:
            del q.page[name]
            q.client.cards.remove(name)


def pipeline_layout(q, i, stage_name, page_name, desc, link=None):
    add_card(q, f'item{i}', ui.wide_info_card(box=ui.box(
        'grid', width='500px'), name=f'#{page_name}', title=stage_name, caption=desc, image=link, label='Learn More',))


def make_markdown_row(values):
    return f"| {' | '.join([str(x) for x in values])} |"


def make_markdown_table(fields, rows):
    return '\n'.join([
        make_markdown_row(fields),
        make_markdown_row('---' * len(fields)),
        '\n'.join([make_markdown_row(row) for row in rows]), ])


@on('#page1')
async def page1(q: Q):
    q.page['sidebar'].value = '#page1'
    # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    clear_cards(q)

    add_card(q, 'title_card', ui.header_card(
        box='vertical',
        title='H2O Olympics | H2O World India Hackathon - Team H2O_U3108',
        subtitle='',
        color='card',
        image='https://wave.h2o.ai/img/h2o-logo.svg'
    ))

    pipeline_layout(q, 1, 'Data', 'page2', 'Observe the training data tables provided',
                    'https://visualstudiomagazine.com/-/media/ECG/visualstudiomagazine/Images/introimages/BigData.jpg')
    pipeline_layout(
        q, 2, 'EDA', 'page3', 'Analyze graphs to gain more insights into data', 'https://thumbs.dreamstime.com/b/business-analytics-isometric-composition-professional-cloud-based-data-insights-online-service-glow-smartphone-screen-vector-148539520.jpg')
    pipeline_layout(q, 3, 'Preprocessing', 'page4',
                    'Based on observations, perform appropriate data preprocessing', 'https://cdn.analyticsvidhya.com/wp-content/uploads/2021/08/584692017-PGHD.jpg')
    pipeline_layout(q, 4, 'Feature Engineering', 'page5',
                    'Leverage existing data to create additional variables to enhance model performance', 'data:image/jpeg;base64,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')
    pipeline_layout(q, 5, 'Model Training', 'page6',
                    'Time to use H2O 3 & Train our Model!', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRQ4Q2JIeuPkrr40vxWO2yZm823hFrwvWTsXZLUuIU8g45ZhiwsYuKl6Q0vsiy0EF7qUJ2EC3hNS3M&usqp=CAU&ec=48600112')
    pipeline_layout(q, 6, 'Model Evaluation', 'page7',
                    'Evaluate the forecasting ability of different models', 'https://miro.medium.com/v2/resize:fit:1200/1*59SfOBEuJ3m0KC7r5SshBQ.png')
    pipeline_layout(q, 7, 'Results', 'page8', 'Here are the final results!',
                    'https://badgeos.org/wp-content/uploads/2018/06/badges_0004_leaderboard-300x300.png')

    # add_card(q, 'stage1', ui.tall_info_card(
    #     box='1 1 4 5',
    #     name='step-1',
    #     title='Data',
    #     caption='Understand the data provided to us',
    #     category='Category',
    # )
    # )


@on('#page2')
async def page2(q: Q):
    q.page['sidebar'].value = '#page2'
    # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    clear_cards(q)

    if q.events.table:
        table = q.page['table_form_card'].items[0].table
        if q.events.table.sort:
            q.client.sort = q.events.table.sort
            q.client.page_offset = 0
        if q.events.table.filter:
            q.client.filters = q.events.table.filter
            q.client.page_offset = 0
        if q.events.table.search is not None:
            q.client.search = q.events.table.search
            q.client.page_offset = 0
        if q.events.table.page_change:
            q.client.page_offset = q.events.table.page_change.get('offset', 0)
        if q.events.table.reset:
            q.client.search = None
            q.client.sort = None
            q.client.filters = None
            q.client.page_offset = 0
            table.pagination = ui.table_pagination(total_rows, rows_per_page)

        offset = q.client.page_offset or 0
        df = get_df(orig_train_df, q.client.sort,
                    q.client.search, q.client.filters)

        if q.events.table.download:
            # Create and upload a CSV file for downloads.
            # For multi-user apps, the tmp file name should be unique for each user, not hardcoded.
            df.to_csv('data_download.csv')
            download_url, = await q.site.upload(['data_download.csv'])
            # Clean up.
            os.remove('data_download.csv')
            q.page['meta'].script = ui.inline_script(
                f'window.open("{download_url}")')

        # Update table pagination according to the new row count.
        if q.client.search is not None or q.client.filters:
            table.pagination = ui.table_pagination(len(df), rows_per_page)

        table.rows = df_to_table_rows(df[offset:offset + rows_per_page])

    add_card(q, 'table_form_card', ui.form_card(box='table_zone', items=[
        ui.table(
            name='table',
            columns=make_table_columns(orig_train_df),
            rows=df_to_table_rows(get_df(orig_train_df)[0:rows_per_page]),
                resettable=True,
                downloadable=True,
                pagination=ui.table_pagination(total_rows, rows_per_page),
        ),
    ])
    )

    add_card(q, 'analysis', ui.form_card(
        box='alert_zone',
        items=[
            ui.message_bar(
                type='info', text='There are 40 unique stations and states combinations.'),
            ui.message_bar(
                type='info', text='Both train and test contain same ids'),
            ui.message_bar(
                type='warning', text='We observe that columns like PM2.5,PM10,O3,CO have null values which need to be handled.'),
            ui.message_bar(
                type='warning', text='Columns like ID_Date,StateCode,Date are non numerical and need to be handled.'),

        ]
    ))


@on('#page3')
async def page3(q: Q):
    q.page['sidebar'].value = '#page3'
    # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    clear_cards(q)

    global Id
    global col_selected
    global is_outlier

    if q.args.id_dropdown:
        val = str(q.args.id_dropdown)[6:]
        Id = int(val)
        print(Id)
    elif q.args.cols_dropdown:
        col_selected = str(q.args.cols_dropdown)
        print(col_selected)

    if q.args.outlier_dropdown:
        x = q.args.outlier_dropdown
        if x == 'Feature Visualization':
            is_outlier = False
        else:
            is_outlier = True

    add_card(q, 'dropdown_outlier_card', ui.form_card(box='horizontal', items=[
        ui.dropdown(name='outlier_dropdown',
                    label='Select graph types:',
                    popup='never',
                    placeholder=f'{current_outlier_selection(is_outlier)}',
                    trigger=True,
                    choices=ui_graph_type_choices()
                    )
    ]))
    add_card(q, 'dropdown_id_card', ui.form_card(box='horizontal', items=[
        ui.dropdown(name='id_dropdown',
                    label='Select location ID value:',
                    popup='never',
                    placeholder=f'ID {Id}',
                    trigger=True,
                    choices=ui_id_list(orig_train_df.copy(), col='ID')
                    )
    ]))
    add_card(q, 'dropdown_cols_card', ui.form_card(box='horizontal', items=[
        ui.dropdown(name='cols_dropdown',
                    label='Select feature variable/target(AQI):',
                    popup='never',
                    placeholder=f'{col_selected}',
                    trigger=True,
                    choices=ui_cols_list(orig_train_df.copy())
                    )
    ]))
    if not is_outlier:
        add_card(q, 'chart_line', ui.plot_card(
            box='vertical',
            title=f'Plot of {col_selected} vs Date for ID {Id}',
            data=data(fields=f'Date {col_selected}',
                      rows=make_col_rows(orig_train_df.copy(), id=Id, col=f'{col_selected}')),
            plot=ui.plot(
                [ui.mark(type='line', x_scale='time', x='=Date', y=f'={col_selected}', y_min=0, color='blue', x_title='Date', y_title=f'{col_selected}')])
        ))
    else:
        add_card(q, 'chart_whisker', ui.image_card(
            box='custom',
            title=f'KDE Plot and Box & Whisker Plot of {col_selected} for ID {Id}',
            image=outlier_detection(
                orig_train_df.copy(), col_selected, 'before'),
            type='png'
        ))

    add_card(q, 'analysis', ui.form_card(
        box='horizontal',
        items=[
            ui.message_bar(
                type='warning', text='Outliers are observed and need to be handled'),
            ui.message_bar(
                type='warning', text='Datapoints are not really continuous as data of some dates are observed to be missing.')
        ]
    ))


@on('#page4')
async def handle_page4(q: Q):
    q.page['sidebar'].value = '#page4'
    # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    # Since this page is interactive, we want to update its card instead of recreating it every time, so ignore 'form' card on drop.
    clear_cards(q, ['icon-stepper'])

    add_card(q, 'title_card', ui.header_card(
        box='vertical',
        title='Preprocessing',
        subtitle='',
        color='card'
    ))

    if q.args.step1:
        # Just update the existing card, do not recreate.
        add_card(q, 'form', ui.form_card(box='vertical', items=[
            ui.stepper(name='stepper', items=[
                ui.step(label='Feature Extraction',
                        icon='DataManagementSettings', done=True),
                ui.step(label='Missing value handling by imputation and interpolation',
                        icon='CopyEdit'),
                ui.step(label='Outlier Handling',
                        icon='AlignHorizontalCenter'),
                ui.step(label='Scaling and Standardization',
                        icon='HistoricalWeather'),
                ui.step(label='Feature Engineering',
                        icon='DatabaseView'),
            ]),
            ui.buttons(justify='center', items=[
                ui.button(name='step0', label='Previous'),
                ui.button(name='step2', label='Next', primary=True),
            ])
        ]))
    elif q.args.step2:
        # Just update the existing card, do not recreate.
        add_card(q, 'form', ui.form_card(box='vertical', items=[
            ui.stepper(name='stepper', items=[
                ui.step(label='Feature Extraction',
                        icon='DataManagementSettings', done=True),
                ui.step(label='Missing value handling by imputation and interpolation',
                        icon='CopyEdit', done=True),
                ui.step(label='Outlier Handling',
                        icon='AlignHorizontalCenter'),
                ui.step(label='Scaling and Standardization',
                        icon='HistoricalWeather'),
                ui.step(label='Feature Engineering',
                        icon='DatabaseView'),
            ]),
            ui.buttons(justify='center', items=[
                ui.button(name='step1', label='Previous'),
                ui.button(name='step3', label='Next', primary=True),
            ])
        ]))
    elif q.args.step3:
        # Just update the existing card, do not recreate.
        add_card(q, 'form', ui.form_card(box='vertical', items=[
            ui.stepper(name='stepper', items=[
                ui.step(label='Feature Extraction',
                        icon='DataManagementSettings', done=True),
                ui.step(label='Missing value handling by imputation and interpolation',
                        icon='CopyEdit', done=True),
                ui.step(label='Outlier Handling',
                        icon='AlignHorizontalCenter', done=True),
                ui.step(label='Scaling and Standardization',
                        icon='HistoricalWeather'),
                ui.step(label='Feature Engineering',
                        icon='DatabaseView'),
            ]),
            ui.buttons(justify='center', items=[
                ui.button(name='step2', label='Previous'),
                ui.button(name='step4', label='Next', primary=True),
            ])
        ]
        ))
    elif q.args.step4:
        # Just update the existing card, do not recreate.
        add_card(q, 'form', ui.form_card(box='vertical', items=[
            ui.stepper(name='stepper', items=[
                ui.step(label='Feature Extraction',
                        icon='DataManagementSettings', done=True),
                ui.step(label='Missing value handling by imputation and interpolation',
                        icon='CopyEdit', done=True),
                ui.step(label='Outlier Handling',
                        icon='AlignHorizontalCenter', done=True),
                ui.step(label='Scaling and Standardization',
                        icon='HistoricalWeather', done=True),
                ui.step(label='Feature Engineering',
                        icon='DatabaseView'),
            ]),
            ui.buttons(justify='center', items=[
                ui.button(name='step3', label='Previous'),
                ui.button(name='step5', label='Next', primary=True),
            ])
        ]
        ))
    elif q.args.step5:
        # Just update the existing card, do not recreate.
        add_card(q, 'form', ui.form_card(box='vertical', items=[
            ui.stepper(name='stepper', items=[
                ui.step(label='Feature Extraction',
                        icon='DataManagementSettings', done=True),
                ui.step(label='Missing value handling by imputation and interpolation',
                        icon='CopyEdit', done=True),
                ui.step(label='Outlier Handling',
                        icon='AlignHorizontalCenter', done=True),
                ui.step(label='Scaling and Standardization',
                        icon='HistoricalWeather', done=True),
                ui.step(label='Feature Engineering',
                        icon='DatabaseView', done=True),
            ]),
            ui.buttons(justify='center', items=[
                ui.button(name='step4', label='Previous'),
                ui.button(name='done', label='Done', primary=True),
            ])
        ]
        ))
    elif q.args.done:
        # Just update the existing card, do not recreate.
        add_card(q, 'form', ui.form_card(box='vertical', items=[
            ui.stepper(name='stepper', items=[
                ui.step(label='Feature Extraction',
                        icon='DataManagementSettings', done=True),
                ui.step(label='Missing value handling by imputation and interpolation',
                        icon='CopyEdit', done=True),
                ui.step(label='Outlier Handling',
                        icon='AlignHorizontalCenter', done=True),
                ui.step(label='Scaling and Standardization',
                        icon='HistoricalWeather', done=True),
                ui.step(label='Feature Engineering',
                        icon='DatabaseView', done=True),
            ]),
            ui.buttons(justify='center', items=[
                ui.button(name='step5', label='Previous'),
            ])
        ]
        ))

        add_card(q, 'table_form_card4', ui.form_card(box='vertical', items=[
            ui.table(
                name='table4',
                columns=make_table_columns(train_df),
                rows=df_to_table_rows(get_df(train_df)[0:rows_per_page]),
                resettable=True,
                downloadable=True,
                pagination=ui.table_pagination(len(train_df), rows_per_page),
            ),
        ])
        )
    else:
        # If first time on this page, create the card.
        add_card(q, 'form', ui.form_card(box='vertical', items=[
            ui.stepper(name='stepper', items=[ui.step(label='Feature Extraction',
                                                      icon='DataManagementSettings'),
                                              ui.step(label='Null value handling by imputation and interpolation',
                                                      icon='CopyEdit'),
                                              ui.step(label='Outlier Handling',
                                                      icon='AlignHorizontalCenter'),
                                              ui.step(label='Scaling and Standardization',
                                                      icon='HistoricalWeather'),
                                              ui.step(label='Feature Engineering',
                                                      icon='DatabaseView'), ]),
            ui.buttons(justify='center', items=[
                ui.button(name='step1', label='Next', primary=True),
            ]),
        ]))


@on('#page5')
async def page5(q: Q):
    q.page['sidebar'].value = '#page5'
    # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    clear_cards(q)

    add_card(q, 'title_card', ui.header_card(
        box='horizontal',
        title='Feature Engineering - AQI Formula Calculation',
        subtitle='',
        color='card'
    ))

    add_card(q, 'aqi_img', ui.form_card(box='custom', items=[
        ui.image(
            title='AQI Calculation',
            path='https://media.springernature.com/lw685/springer-static/image/chp%3A10.1007%2F978-3-030-54765-3_9/MediaObjects/494075_1_En_9_Fig1_HTML.png',
        ),
    ]))

    add_card(q, 'new_feature_info', ui.tall_info_card(
        box='vertical',
        name='info_card',
        title='Air Quality Index Calculation Methodology',
        caption='''
            - The AQI calculation uses 7 measures: PM2.5, PM10, SO2, NOx, NH3, CO and O3.
            - For PM2.5, PM10, SO2, NOx and NH3 the average value in last 24-hrs is used with the condition of having at least 16 values.
            - For CO and O3 the maximum value in last 8-hrs is used.
            - Each measure is converted into a Sub-Index based on pre-defined groups.
            - Sometimes measures are not available due to lack of measuring or lack of required data points.
            - Final AQI is the maximum Sub-Index with the condition that at least one of PM2.5 and PM10 should be available and at least three out of the seven should be available.''',
        category='',
        icon='LightBulb',
        image=r'https://i.postimg.cc/ZRxJ9v04/AQI.jpg',
        image_height='200px'
    ))


@on('#page6')
async def page6(q: Q):
    q.page['sidebar'].value = '#page6'
    # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    clear_cards(q)


@on('#page7')
async def page7(q: Q):
    q.page['sidebar'].value = '#page7'
    # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    clear_cards(q)

    add_card(q, 'title_card', ui.header_card(
        box='horizontal',
        title='List of All Models',
        subtitle='',
        color='card'
    ))


async def init(q: Q) -> None:
    q.page['meta'] = ui.meta_card(title='Air Foresight',
                                  box='', layouts=[ui.layout(breakpoint='xs', min_height='100vh', zones=[
                                      ui.zone('main', size='1', direction=ui.ZoneDirection.ROW, zones=[
                                          ui.zone('sidebar', size='250px'),
                                          ui.zone('body', zones=[
                                              ui.zone('header'),
                                              ui.zone('content', zones=[
                                                  # Specify various zones and use the one that is currently needed. Empty zones are ignored.
                                                  ui.zone(
                                                      'horizontal', direction=ui.ZoneDirection.ROW),
                                                  ui.zone('horizontal_diff', direction=ui.ZoneDirection.ROW, zones=[
                                                      ui.zone(
                                                          'table_zone', size='50%',),
                                                      ui.zone(
                                                          'alert_zone', size='50%',)
                                                  ]),
                                                  ui.zone('vertical'),
                                                  ui.zone('grid', direction=ui.ZoneDirection.ROW,
                                                          wrap='stretch', justify='center'),
                                                  ui.zone('custom',
                                                          wrap='center', size='80%'),
                                                  ui.zone('button_start',
                                                          wrap='start'),
                                              ]),
                                          ]),
                                      ])
                                  ])], stylesheet=ui.inline_stylesheet('''
div[data-test="body"]>div {
   background-image: url("https://s01.sgp1.cdn.digitaloceanspaces.com/article/182256-jmoxpoqxcl-1667446898.jpg");
   background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}
'''))
    q.page['sidebar'] = ui.nav_card(
        box='sidebar', color='card', title='', subtitle="Predicting the air you breathe.",
        value=f'#{q.args["#"]}' if q.args['#'] else '#page1',
        image='https://i.postimg.cc/tJc7XTC9/Air-Foresight.png', items=[
            ui.nav_group('Explore', items=[
                ui.nav_item(name='#page1', label='Home'),
                ui.nav_item(name='#page2', label='Data Tables'),
                ui.nav_item(name='#page3', label='EDA Charts'),
                ui.nav_item(name='#page4', label='Preprocessing'),
                ui.nav_item(name='#page5', label='Feature Engineering'),
                ui.nav_item(name='#page6', label='Model Training'),
                ui.nav_item(name='#page7', label='Model Evaluation'),
                ui.nav_item(name='#page8', label='Results'),
            ]),
        ])
    # If no active hash present, render page1.
    if q.args['#'] is None:
        await page1(q)


@app('/')
async def serve(q: Q):
    # Run only once per client connection.
    if not q.client.initialized:
        q.client.cards = set()
        await init(q)
        q.client.initialized = True
    # Handle routing.
    await handle_on(q)
    await q.page.save()
