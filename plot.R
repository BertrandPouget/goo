rm(list=ls())
graphics.off()

library(sf)
library(sp)
library(ggplot2)

load('data/r/data.RData')
geo = st_read('data/r/geo.shx')
results = scan(file='results.txt',what=numeric(),sep='\t')

ttt = character(length(results))
ttt[which(results==0)]="0. paved"
ttt[which(results==1)]="1. unpaved"
ttt[which(results==2)]="2. uncertain"

windows();  ggplot() + 
  geom_sf(data = geo, aes(color=ttt,fill=ttt))+
  scale_fill_manual(values=c("blue", "brown", "purple"))+
  scale_color_manual(values=c("blue", "brown", "purple"))+
  labs(fill= "Pavement surface")+
  coord_sf() +
  theme(panel.grid.major = element_line(color = gray(.9), linetype=3, size=0.2), 
        panel.background = element_rect(fill="cornsilk1"))+
  guides(color=FALSE)
