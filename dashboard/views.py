from django.shortcuts import render

from .ai_model import predict_video

import pandas as pd
import plotly.express as px


def dashboard_view(request):


    prediction_result = predict_video(

        "dashboard/static/videos/demo4.mp4"
    )

    prediction = prediction_result["label"]

    confidence = prediction_result["confidence"]


    alerts = []

    if prediction == "high-risk":

        alerts.append(
            "Violence detected in surveillance stream"
        )

        alerts.append(
            "High-risk crowd behavior detected"
        )

        alerts.append(
            "Critical anomaly detected"
        )

    elif prediction == "suspicious":

        alerts.append(
            "Suspicious activity detected"
        )

        alerts.append(
            "Potential abnormal movement"
        )

    else:

        alerts.append(
            "No abnormal activity detected"
        )


    glotip_df = pd.read_excel(

        "dashboard/data/data_glotip.xlsx",

        header=2
    )


    indicators = [

        "Detected trafficking victims",

        "Offences of trafficking in persons",

        "Persons brought into formal contact",

        "Persons prosecuted"
    ]

    risk_data = glotip_df[

        glotip_df["Indicator"].isin(
            indicators
        )

    ].copy()


    risk_data["txtVALUE"] = risk_data[
        "txtVALUE"
    ].replace("<5", 2)

    risk_data["txtVALUE"] = pd.to_numeric(

        risk_data["txtVALUE"],

        errors="coerce"
    )

    risk_data["Year"] = pd.to_numeric(

        risk_data["Year"],

        errors="coerce"
    )

    risk_data = risk_data.dropna(

        subset=["txtVALUE", "Year"]
    )

    risk_data["Year"] = risk_data[
        "Year"
    ].astype(int)


    country_risk = risk_data.groupby(

        ["Iso3_code", "Country"]

    )["txtVALUE"].sum().reset_index()

    country_risk.columns = [

        "country_code",

        "country_name",

        "risk_score"
    ]


    map_data = risk_data.groupby(

        ["Iso3_code", "Country", "Year"]

    )["txtVALUE"].sum().reset_index()

    map_data = map_data.sort_values(
        "Year"
    )


    fig = px.choropleth(

    map_data,

    locations="Iso3_code",

    locationmode="ISO-3",

    color="txtVALUE",

    labels={

        "txtVALUE": "Risk Score"
    },

    hover_name="Country",

    animation_frame="Year",

    color_continuous_scale="Reds",

    range_color=(

        0,

        map_data["txtVALUE"].max()
    ),

    title="Global Risk Distribution Over Time"
    )

    fig.update_layout(

        width=1200,

        height=650,

        margin=dict(

            l=0,
            r=0,
            t=50,
            b=0
        )
    )
    fig.update_coloraxes(

        colorbar_title="Risk Score"
    )

    map_html = fig.to_html()


    yearly_data = risk_data.groupby(

        "Year"

    )["txtVALUE"].sum().reset_index()

    timeline_fig = px.line(

        yearly_data,

        x="Year",

        y="txtVALUE",

        markers=True,

        title="Detected Events by Year"
    )

    timeline_fig.update_layout(

        height=350,

        margin=dict(

            l=0,
            r=0,
            t=40,
            b=0
        )
    )

    timeline_html = timeline_fig.to_html()


    total_events = len(risk_data)

    countries_monitored = len(
        country_risk
    )


    context = {

        "prediction": prediction,

        "confidence": confidence,

        "alerts": alerts,

        "map_html": map_html,

        "timeline_html": timeline_html,

        "total_events": total_events,

        "countries_monitored": countries_monitored,
    }

    return render(

        request,

        "dashboard/index.html",

        context
    )