import plotly.graph_objects as go

fig = go.Figure()

# Add bars with individual colors
fig.add_trace(go.Bar(
    x=[
        "Fully correct recall",
        "Incorrect recall"
    ],
    y=[25, 6],
    marker_color=["#2ecc71", "#e74c3c"],
    text=["27 (87.1%)", "4 (12.9%)"],
    textposition='outside',
    showlegend=False
))

fig.update_layout(
    title="Five-Second Test Recall Performance (n = 31)",
    yaxis_title="Number of participants",
    xaxis_title="",
    yaxis=dict(range=[0, 30])
)

# Save the figure as PNG in assets folder
output_path = "/Users/maximilianwilke/geant-rag/assets/recall_performance.png"
fig.write_image(output_path, width=800, height=600, scale=2)
print(f"Chart saved to: {output_path}")

fig.show()