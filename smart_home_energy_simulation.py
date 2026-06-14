import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': '#0A0D13','axes.facecolor':'#111520',
    'axes.edgecolor':'#1E2535','axes.labelcolor':'#CBD5E1',
    'xtick.color':'#64748B','ytick.color':'#64748B',
    'grid.color':'#1E2535','grid.alpha':0.6,'text.color':'#E2E8F0',
    'font.family':'DejaVu Sans','axes.titlepad':14,
    'axes.titlesize':12,'axes.titleweight':'bold',
})
YELLOW,BLUE,GREEN,RED,ORANGE,PURPLE = '#FACC15','#38BDF8','#4ADE80','#F87171','#FB923C','#C084FC'

np.random.seed(42)
timestamps = pd.date_range('2024-06-01', periods=30*24, freq='1h')
n = len(timestamps)
hours  = timestamps.hour.to_numpy()
days   = (timestamps - timestamps[0]).days.to_numpy()
weekday = timestamps.dayofweek.to_numpy()  # 0=Mon,6=Sun

# ── APPLIANCE ENERGY CONSUMPTION SIMULATION ──────────────────
# AC: heavy during day (9am-11pm), more on weekends
ac = np.where((hours>=9)&(hours<=23), 1.8, 0.0)
ac += np.where(weekday>=5, 0.5, 0)               # weekend boost
ac += np.random.normal(0, 0.15, n)
ac = np.clip(ac, 0, 3.5)

# Washing Machine: used mornings & evenings (Mon-Sat)
wm = np.where(((hours>=8)&(hours<=10))|((hours>=17)&(hours<=19)), 0.9, 0.0)
wm *= np.where(weekday<6, 1, 0.3)
wm += np.random.normal(0, 0.05, n)
wm = np.clip(wm, 0, 1.5)

# Refrigerator: always on, slight day/night cycle
fridge = 0.18 + 0.03*np.sin(2*np.pi*hours/24) + np.random.normal(0,0.01,n)
fridge = np.clip(fridge, 0.12, 0.28)

# Lighting: evening hours, less during day
lighting = np.where((hours>=18)&(hours<=22), 0.35, 0.0)
lighting += np.where((hours>=6)&(hours<=8), 0.15, 0.0)
lighting += np.random.normal(0, 0.02, n)
lighting = np.clip(lighting, 0, 0.6)

# EV Charger: overnight charging (10pm-5am), only some days
ev_days = np.isin(days % 3, [0])                 # every 3rd day
ev = np.where((hours>=22)|(hours<=5), 3.2, 0.0) * ev_days
ev += np.random.normal(0, 0.1, n)
ev = np.clip(ev, 0, 4.0)

# Solar generation: daytime only
solar = np.maximum(0, 2.8*np.sin(np.pi*(hours-6)/12))
solar += np.random.normal(0, 0.15, n)
solar *= np.where(days%7 == 3, 0.4, 1.0)         # cloudy day every week
solar = np.clip(solar, 0, 4.0)

total_consumption = ac + wm + fridge + lighting + ev
net_grid = total_consumption - solar               # positive=buying, negative=selling

# Daily electricity cost (₹8/unit)
RATE = 8.0
cost_hourly = np.where(net_grid > 0, net_grid*RATE, 0)

df = pd.DataFrame({
    'timestamp': timestamps,
    'ac_kw': ac.round(3),
    'washing_machine_kw': wm.round(3),
    'refrigerator_kw': fridge.round(3),
    'lighting_kw': lighting.round(3),
    'ev_charger_kw': ev.round(3),
    'total_consumption_kw': total_consumption.round(3),
    'solar_generation_kw': solar.round(3),
    'net_grid_kw': net_grid.round(3),
    'cost_inr': cost_hourly.round(2),
})
df['date']    = df['timestamp'].dt.date
df['hour']    = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.day_name()
df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5

# ── ALERTS ────────────────────────────────────────────────────
THRESH_KW = 4.5
df['alert_high_consumption'] = (df['total_consumption_kw'] > THRESH_KW).astype(int)
df['alert_solar_low']        = ((hours>=10)&(hours<=14)&(df['solar_generation_kw']<0.5)).astype(int)
df['any_alert'] = df[['alert_high_consumption','alert_solar_low']].max(axis=1)

df.to_csv('energy_data.csv', index=False)
total_consumed = df['total_consumption_kw'].sum()
total_solar    = df['solar_generation_kw'].sum()
total_cost     = df['cost_inr'].sum()
solar_pct      = total_solar/total_consumed*100

print(f"✅ Generated {len(df)} hourly readings over 30 days")
print(f"⚡ Total Consumed : {total_consumed:.0f} kWh")
print(f"☀️  Solar Generated: {total_solar:.0f} kWh ({solar_pct:.1f}% of demand)")
print(f"💰 Total Cost     : ₹{total_cost:,.0f}")
print(f"⚠️  High-load alerts: {df['alert_high_consumption'].sum()}")

# ── CHART 1 — MAIN ENERGY DASHBOARD ──────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#0A0D13')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

# Total consumption + solar (sample: first 7 days)
week = df[df['timestamp'] < df['timestamp'][0] + pd.Timedelta(days=7)]
ax1 = fig.add_subplot(gs[0,:])
ax1.set_facecolor('#111520')
ax1.plot(week['timestamp'], week['total_consumption_kw'], color=ORANGE, linewidth=1.6, label='Total Consumption (kW)', alpha=0.9)
ax1.fill_between(week['timestamp'], week['total_consumption_kw'], alpha=0.1, color=ORANGE)
ax1.plot(week['timestamp'], week['solar_generation_kw'], color=YELLOW, linewidth=1.6, label='Solar Generation (kW)', alpha=0.9)
ax1.fill_between(week['timestamp'], week['solar_generation_kw'], alpha=0.1, color=YELLOW)
ax1.axhline(y=THRESH_KW, color=RED, linestyle='--', linewidth=1.2, alpha=0.6, label=f'High Load Threshold ({THRESH_KW} kW)')
high = week[week['alert_high_consumption']==1]
ax1.scatter(high['timestamp'], high['total_consumption_kw'], color=RED, s=30, zorder=5, label='High Load Alert')
ax1.set_title('⚡  Week 1: Energy Consumption vs Solar Generation', fontsize=13, color='#E2E8F0')
ax1.set_ylabel('Power (kW)')
ax1.legend(framealpha=0.15, labelcolor='white', fontsize=9, ncol=4)
ax1.grid(True, alpha=0.25)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='x', rotation=20)

# Appliance breakdown (avg hourly kW)
ax2 = fig.add_subplot(gs[1,0]); ax2.set_facecolor('#111520')
appliances = ['ac_kw','washing_machine_kw','refrigerator_kw','lighting_kw','ev_charger_kw']
app_labels  = ['AC','Washing\nMachine','Refrigerator','Lighting','EV Charger']
app_colors  = [BLUE, PURPLE, GREEN, YELLOW, ORANGE]
app_avgs    = [df[a].mean()*24 for a in appliances]   # daily kWh
bars = ax2.bar(app_labels, app_avgs, color=app_colors, alpha=0.85, edgecolor='none', width=0.6)
ax2.set_title('🏠  Daily Energy Usage by Appliance (kWh)')
ax2.set_ylabel('kWh/day')
ax2.grid(True, axis='y', alpha=0.25)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
for b, v in zip(bars, app_avgs):
    ax2.text(b.get_x()+b.get_width()/2, v+0.05, f'{v:.1f}', ha='center', color='#E2E8F0', fontsize=10, fontweight='bold')

# Daily cost trend
daily = df.groupby('date').agg(
    total_kwh=('total_consumption_kw','sum'),
    solar_kwh=('solar_generation_kw','sum'),
    cost_inr=('cost_inr','sum'),
    alerts=('any_alert','sum')
).reset_index()
ax3 = fig.add_subplot(gs[1,1]); ax3.set_facecolor('#111520')
ax3.bar(range(len(daily)), daily['cost_inr'],
        color=[RED if c>daily['cost_inr'].quantile(0.75) else GREEN for c in daily['cost_inr']],
        alpha=0.85, edgecolor='none')
ax3.axhline(y=daily['cost_inr'].mean(), color=YELLOW, linestyle='--', linewidth=1.3, alpha=0.7,
            label=f"Avg ₹{daily['cost_inr'].mean():.0f}/day")
ax3.set_title('💰  Daily Electricity Cost (₹)')
ax3.set_ylabel('Cost (₹)'); ax3.set_xlabel('Day')
ax3.legend(framealpha=0.15, labelcolor='white', fontsize=9)
ax3.grid(True, axis='y', alpha=0.25)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

fig.suptitle('🏠  Smart Home Energy Monitoring Dashboard — 30-Day Overview', fontsize=15, color='#E2E8F0', y=1.01)
plt.savefig('chart1_energy_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0A0D13')
plt.close(); print("Saved chart1")

# ── CHART 2 — HOURLY HEATMAP + APPLIANCE PIE ─────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Hourly heatmap (consumption by hour × weekday)
df['weekday_short'] = df['timestamp'].dt.strftime('%a')
weekday_order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
pivot = df.pivot_table(values='total_consumption_kw', index='weekday_short', columns='hour', aggfunc='mean')
pivot = pivot.reindex(weekday_order)
sns.heatmap(pivot, annot=False, cmap='YlOrRd', linewidths=0.2, linecolor='#0A0D13',
            cbar_kws={'label':'Avg kW'}, ax=ax1)
ax1.set_title('🗓️  Avg Consumption: Weekday × Hour')
ax1.set_xlabel('Hour of Day'); ax1.set_ylabel('')

# Appliance share donut
app_totals = [df[a].sum() for a in appliances]
wedges, texts, autotexts = ax2.pie(
    app_totals, labels=app_labels, autopct='%1.1f%%',
    colors=app_colors, startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='#0A0D13', linewidth=1.5),
    textprops={'color':'#CBD5E1','fontsize':9}, pctdistance=0.78)
for at in autotexts: at.set_color('#0A0D13'); at.set_fontweight('bold')
ax2.set_title('🥧  Appliance Consumption Share')

plt.tight_layout()
plt.savefig('chart2_heatmap_appliance.png', dpi=150, bbox_inches='tight', facecolor='#0A0D13')
plt.close(); print("Saved chart2")

# ── CHART 3 — SOLAR SAVINGS + WEEKLY COMPARISON ──────────────
daily['solar_savings_inr'] = daily['solar_kwh'] * RATE
daily['net_cost_inr'] = daily['cost_inr']
weekly = daily.copy()
weekly['week'] = pd.to_datetime(daily['date']).dt.isocalendar().week.values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

weeks = range(len(daily))
ax1.bar(weeks, daily['total_kwh'], label='Total Consumed (kWh)', color=ORANGE, alpha=0.75, edgecolor='none')
ax1.bar(weeks, daily['solar_kwh'], label='Solar Generated (kWh)', color=YELLOW, alpha=0.85, edgecolor='none')
ax1.set_title('☀️  Daily Consumption vs Solar Generation')
ax1.set_xlabel('Day'); ax1.set_ylabel('kWh')
ax1.legend(framealpha=0.15, labelcolor='white', fontsize=9)
ax1.grid(True, axis='y', alpha=0.25)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

ax2.bar(weeks, daily['solar_savings_inr'], color=GREEN, alpha=0.85, edgecolor='none', label='Solar Savings (₹)')
ax2.plot(weeks, daily['cost_inr'], color=RED, linewidth=2, label='Grid Cost (₹)', marker='o', markersize=3)
ax2.set_title('💰  Solar Savings vs Grid Cost per Day')
ax2.set_xlabel('Day'); ax2.set_ylabel('₹')
ax2.legend(framealpha=0.15, labelcolor='white', fontsize=9)
ax2.grid(True, axis='y', alpha=0.25)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('chart3_solar_savings.png', dpi=150, bbox_inches='tight', facecolor='#0A0D13')
plt.close(); print("Saved chart3")

# ── CHART 4 — ALERT SUMMARY ───────────────────────────────────
alert_counts = {
    'High Load (>6kW)': int(df['alert_high_consumption'].sum()),
    'Low Solar (10-14h)': int(df['alert_solar_low'].sum()),
}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
names = list(alert_counts.keys()); vals = list(alert_counts.values())
bars = ax1.barh(names, vals, color=[RED, YELLOW], alpha=0.85, edgecolor='none')
ax1.set_title('🚨  Alert Summary (30 Days)')
ax1.set_xlabel('Number of Hours')
ax1.grid(True, axis='x', alpha=0.25)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
for bar, val in zip(bars, vals):
    ax1.text(val+0.5, bar.get_y()+bar.get_height()/2, str(val), va='center', color='#E2E8F0', fontsize=11)

status = {'Normal Usage':(df['any_alert']==0).sum(),'Alert Triggered':(df['any_alert']==1).sum()}
wedges, texts, autotexts = ax2.pie(
    status.values(), labels=status.keys(), autopct='%1.1f%%',
    colors=[GREEN, RED], startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='#0A0D13', linewidth=2),
    textprops={'color':'#E2E8F0','fontsize':10})
for at in autotexts: at.set_color('#0A0D13'); at.set_fontweight('bold')
ax2.set_title('🛡️  System Health Overview')
plt.tight_layout()
plt.savefig('chart4_alerts.png', dpi=150, bbox_inches='tight', facecolor='#0A0D13')
plt.close(); print("Saved chart4")

# ── LIVE MONITORING ───────────────────────────────────────────
print("\n" + "="*75)
print("  LIVE ENERGY MONITORING — LAST 10 READINGS")
print("="*75)
print(f"{'Time':<12} {'Total':>7} {'Solar':>7} {'Net Grid':>9} {'Cost(₹)':>9}  {'Alert'}")
print("-"*75)
for _, row in df.tail(10).iterrows():
    alert_str = "⚠️ HIGH LOAD" if row['alert_high_consumption'] else "✅ OK"
    grid_str  = f"{row['net_grid_kw']:+.2f}"
    print(f"{row['timestamp'].strftime('%d %b %H:%M'):<12} {row['total_consumption_kw']:>6.2f}kW "
          f"{row['solar_generation_kw']:>6.2f}kW {grid_str:>9}kW ₹{row['cost_inr']:>7.2f}  {alert_str}")
print("="*75)

# ── FINAL REPORT ──────────────────────────────────────────────
best_day = daily.loc[daily['cost_inr'].idxmin(), 'date']
worst_day = daily.loc[daily['cost_inr'].idxmax(), 'date']
total_solar_savings = (daily['solar_savings_inr']).sum()

print()
print("╔══════════════════════════════════════════════════════╗")
print("║   SMART HOME ENERGY MONITOR — FINAL REPORT          ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  📅 Period          : 30 Days (720 hourly readings) ║")
print(f"║  ⚡ Total Consumed  : {total_consumed:.0f} kWh{'':<19}║")
print(f"║  ☀️  Solar Generated : {total_solar:.0f} kWh ({solar_pct:.1f}% coverage){'':<5}║")
print(f"║  💰 Total Grid Cost : ₹{total_cost:>8,.0f}{'':<17}║")
print(f"║  💚 Solar Savings   : ₹{total_solar_savings:>8,.0f}{'':<17}║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  📉 Cheapest Day   : {str(best_day):<29}║")
print(f"║  📈 Costliest Day  : {str(worst_day):<29}║")
print(f"║  🔋 Top Appliance  : AC (highest daily usage)       ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  🚨 High-Load Alerts: {int(df['alert_high_consumption'].sum()):<28}║")
print(f"║  ✅ Normal Hours   : {int((df['any_alert']==0).sum()):<28}║")
print("╠══════════════════════════════════════════════════════╣")
print("║  📁 Files Saved:                                     ║")
print("║     energy_data.csv                                 ║")
print("║     chart1_energy_dashboard.png                     ║")
print("║     chart2_heatmap_appliance.png                    ║")
print("║     chart3_solar_savings.png                        ║")
print("║     chart4_alerts.png                               ║")
print("╚══════════════════════════════════════════════════════╝")
