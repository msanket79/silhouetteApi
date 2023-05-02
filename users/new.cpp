#include<bits/stdc++.h>
using namespace std;
struct process{
	int pid;
	int at;
	int bt;
	int rt;
	int tt;
	int wt;
};
int main()
{
	int n;
	int quantum;
	int avg_waiting_time=0,avg_turnaround_time=0;
	cout<<"enter the no of processsess"<<endl;
	cin>>n;
	cout<<"enter the quantum time"<<endl;
	cin>>quantum;
	queue<process> 	q;
	process p[n];
	cout<<"enter the pid the at, and the burst time of the processes"<<endl;
	for(int i=0;i<n;i++){
		cout<<"hello"<<endl;
		cin>>p[i].pid>>p[i].at>>p[i].bt;
		p[i].rt=p[i].bt;
	
	}
	cout<<"dsd";
	int curr_time=0;
	int flag=0;
	do{
		//flag=0;
		for(int i=0;i<n;i++){
	
			if(p[i].at<=curr_time && (p[i].rt>0)){
				q.push(p[i]);
			}
			//if(p[i].rt>0) flag=1;
		}
		if(q.empty()){
			curr_time++;
			continue;
		}
		process curr=q.front();
		q.pop();
		int et=min(curr.rt,quantum);
		curr.rt-=et;
		cout<<curr.pid<<"  "<<curr_time<<"  "<<curr_time+et<<endl;
		curr_time+=et;
		if (curr.rt>0){
		 q.push(curr);
		 }
		else{
			curr.tt=curr_time-curr.at;
			curr.wt=curr.tt-curr.bt;
			avg_waiting_time+=curr.wt;
			avg_turnaround_time+=curr.tt;
			}
	
		} while(!q.empty());
		return 0;
}


