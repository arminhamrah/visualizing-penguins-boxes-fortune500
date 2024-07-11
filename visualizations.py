#make sure to download the seaborn, pandas, and numpy SDKs beforehand

import numpy as np
import pandas as pd
import seaborn as sns

#penguins
sns.set_theme(style="whitegrid")
penguins = sns.load_dataset("penguins")

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=penguins, kind="bar",
    x="species", y="body_mass_g", hue="sex",
    errorbar="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Body mass (g)")
g.legend.set_title("")

#dots
dots = sns.load_dataset("dots")
palette = sns.color_palette("rocket_r")

sns.relplot(
    data=dots,
    x="time", y="firing_rate",
    hue="coherence", size="choice", col="align",
    kind="line", size_order=["T1", "T2"], palette=palette,
    height=15, aspect=1, facet_kws=dict(sharex=False),
)

#brain networks
df = sns.load_dataset("brain_networks", header=[0, 1, 2, 3, 4], index_col=0)

used_networks = [1, 4, 7, 8, 11, 14, 15, 17]
used_columns = (df.columns
                  .get_level_values("network")
                  .astype(int)
                  .isin(used_networks))
df = df.loc[:, used_columns]

df.columns = df.columns.map("-".join)

corr_mat = df.corr().stack().reset_index(name="correlation")

g = sns.relplot(
    data=corr_mat,
    x="level_0", y="level_1", hue="correlation", size="correlation",
    palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    height=10, sizes=(50, 250), size_norm=(-.2, .8),
)

g.set(xlabel="", ylabel="", aspect="equal")
g.despine(left=True, bottom=True)
g.ax.margins(.02)
for label in g.ax.get_xticklabels():
    label.set_rotation(90)

#ribbon boxes ;)
# the ribbon box example from 
#     https://matplotlib.org/stable/gallery/misc/demo_ribbon_box.html

import numpy as np

from matplotlib import cbook, colors as mcolors
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransformTo


class RibbonBox:

    original_image = plt.imread(
        cbook.get_sample_data("Minduka_Present_Blue_Pack.png"))
    cut_location = 70
    b_and_h = original_image[:, :, 2:3]
    color = original_image[:, :, 2:3] - original_image[:, :, 0:1]
    alpha = original_image[:, :, 3:4]
    nx = original_image.shape[1]

    def __init__(self, color):
        rgb = mcolors.to_rgb(color)
        self.im = np.dstack(
            [self.b_and_h - self.color * (1 - np.array(rgb)), self.alpha])

    def get_stretched_image(self, stretch_factor):
        stretch_factor = max(stretch_factor, 1)
        ny, nx, nch = self.im.shape
        ny2 = int(ny*stretch_factor)
        return np.vstack(
            [self.im[:self.cut_location],
             np.broadcast_to(
                 self.im[self.cut_location], (ny2 - ny, nx, nch)),
             self.im[self.cut_location:]])


class RibbonBoxImage(AxesImage):
    zorder = 1

    def __init__(self, ax, bbox, color, *, extent=(0, 1, 0, 1), **kwargs):
        super().__init__(ax, extent=extent, **kwargs)
        self._bbox = bbox
        self._ribbonbox = RibbonBox(color)
        self.set_transform(BboxTransformTo(bbox))

    def draw(self, renderer, *args, **kwargs):
        stretch_factor = self._bbox.height / self._bbox.width

        ny = int(stretch_factor*self._ribbonbox.nx)
        if self.get_array() is None or self.get_array().shape[0] != ny:
            arr = self._ribbonbox.get_stretched_image(stretch_factor)
            self.set_array(arr)

        super().draw(renderer, *args, **kwargs)


def main():
    fig, ax = plt.subplots()

    years = np.arange(2004, 2009)
    heights = [7900, 8100, 7900, 6900, 2800]
    box_colors = [
        (0.8, 0.2, 0.2),
        (0.2, 0.8, 0.2),
        (0.2, 0.2, 0.8),
        (0.7, 0.5, 0.8),
        (0.3, 0.8, 0.7),
    ]

    for year, h, bc in zip(years, heights, box_colors):
        bbox0 = Bbox.from_extents(year - 0.4, 0., year + 0.4, h)
        bbox = TransformedBbox(bbox0, ax.transData)
        ax.add_artist(RibbonBoxImage(ax, bbox, bc, interpolation="bicubic"))
        ax.annotate(str(h), (year, h), va="bottom", ha="center")

    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_ylim(0, 10000)

    background_gradient = np.zeros((2, 2, 4))
    background_gradient[:, :, :3] = [1, 1, 0]
    background_gradient[:, :, 3] = [[0.1, 0.3], [0.3, 0.5]]  # alpha channel
    ax.imshow(background_gradient, interpolation="bicubic", zorder=0.1,
              extent=(0, 1, 0, 1), transform=ax.transAxes, aspect="auto")

    plt.show()


main()

#most lucrative companies, 1955-2005
import pandas as pd
import matplotlib.pyplot as plt

#read in the data
df = pd.read_csv('fortune500.csv')

#print range of years in dataset
print(f"Years present in the dataset: {df['Year'].min()} to {df['Year'].max()}")

#year -> integer
df['Year'] = df['Year'].astype(int)

#filter based on year
df_filtered = df[(df['Year'] >= 1955) & (df['Year'] <= 2005)]

#make sure no extra data after filtering
if df_filtered.empty:
    print("No data available for the years 1955-2005.")
    print("Available years in the dataset:")
    print(df['Year'].unique())
else:
    df_filtered['Profit (in millions)'] = pd.to_numeric(df_filtered['Profit (in millions)'].replace('N.A.', pd.NA), errors='coerce') #profit -> float
    top_companies = df_filtered.loc[df_filtered.groupby('Year')['Profit (in millions)'].idxmax()] #group by year & find best of each year
    top_companies = top_companies.sort_values('Year') #sort by year
    plt.figure(figsize=(15, 8)) #create bar plot
    bars = plt.bar(top_companies['Year'], top_companies['Profit (in millions)'])

    #customize plot
    plt.title('Most Lucrative Company Each Year (1955-2005)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Profit (in millions)', fontsize=12)
    plt.xticks(top_companies['Year'], rotation=45)

    #add company names
    for bar in bars:
        height = bar.get_height()
        year = int(bar.get_x())
        company = top_companies.loc[top_companies['Year'] == year, 'Company']
        if not company.empty:
            company_name = company.values[0]
            plt.text(bar.get_x() + bar.get_width()/2., height + (plt.ylim()[1] - plt.ylim()[0]) * 0.01,  # Add small gap
                 company_name,
                 ha='center', va='bottom', rotation=90, fontsize=8)

plt.tight_layout()
plt.show()

#all these graphs should just open up on your computer as images -- enjoy, and feel free to further tinker around!