#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<cstring>
#include<cassert>
#include<cstdlib>
#include<vector>
#include<algorithm>
#include<stack>
#include<sqlite3.h>
 
using namespace std;
 
const int vec_len=1618;     //一共vec_len个基本义原
const double alpha=1.6;     //计算基本义原相似度时的参数
const double beta1=0.5;     //4种描述式相似度的权值
const double beta2=0.2;
const double beta3=0.17;
const double beta4=0.13;
//const double delta=0.2;
//const double gama=0.2;
 
class myclass{
public:
    int index;
    string sememe;
    int parent;
    myclass(){};
    myclass(int i,string sem,int p):index(i),sememe(sem),parent(p){}
    //重载关系运算符是为了使用STL中的find()函数
    inline bool operator == (const myclass & m){
        return sememe.compare(m.sememe)==0;
    }
    inline bool operator >(const myclass & m) const {
        return sememe.compare(m.sememe)>0;
    }
    inline bool operator <(const myclass & m) const {
        return sememe.compare(m.sememe)<0;
    }
};
 
vector<myclass> semeVec(vec_len);
 
//把基本义原从文件读入vector
void initSemVec(string filename){
    ifstream ifs(filename.c_str());
    assert(ifs);
    string line;
    while(getline(ifs,line)){
        istringstream stream(line);
        int index,pind;
        string seme;
        stream>>index>>seme>>pind;
        myclass mc(index,seme,pind);
        semeVec[index]=mc;
    }
    ifs.close();
}
 
//把用逗号分隔的string片段放到vector<string>中
static void splitString(string line,vector<string> &vec){
    string::size_type pos1,pos2;
    pos1=0;
    while((pos2=line.find(",",pos1))!=string::npos){
        string sem=line.substr(pos1,pos2-pos1);
        //把可能包含的一对小括号去掉
        /*string::size_type pp=sem.find("(");
        if(pp!=string::npos){
            sem.erase(pp,1);
            pp=sem.find(")");
            sem.erase(pp,1);
        }*/
        vec.push_back(sem);
        pos1=pos2+1;
        if(pos1>line.size())
            break;
    }
}
 
//计算两个基本义原的相似度
double calSimBase(string sem1,string sem2){
    assert(sem1.size()>0 && sem2.size()>0);
    if(sem1[0]==40 ^ sem2[0]==40)       //有一个是具体词，而另一个不是
        return 0;
    if(sem1[0]==40 && sem2[0]==40){     //如果两个都是具体词
        if(sem1!=sem2)
            return 0.0;
    }
    if(sem1==sem2)
        return 1.0;
    cout<<"将要计算基本义原["<<sem1<<"]和["<<sem2<<"]的相似度"<<endl;
    stack<string> stk1,stk2;
    myclass mc1(0,sem1,0);
    myclass mc2(0,sem2,0);
    vector<myclass>::iterator itr=find(semeVec.begin(),semeVec.end(),mc1);
    if(itr==semeVec.end()){
        cout<<"["<<sem1<<"]不在词典中"<<endl;
        return 0;
    }
    //把sem1的路径压入栈中
    stk1.push(sem1);
    int child=itr->index;
    int parent=itr->parent;
    while(child!=parent){
        stk1.push(semeVec[parent].sememe);
        child=parent;
        parent=semeVec[parent].parent;
    }
     
    itr=find(semeVec.begin(),semeVec.end(),mc2);
    if(itr==semeVec.end()){
        cout<<"["<<sem2<<"]不在词典中"<<endl;
        return 0;
    }
    //把sem2的路径压入栈中
    stk2.push(sem2);
    child=itr->index;
    parent=itr->parent;
    while(child!=parent){
        stk2.push(semeVec[parent].sememe);
        child=parent;
        parent=semeVec[parent].parent;
    }
     
    if(stk1.top()!=stk2.top()){
        cout<<"["<<stk1.top()<<"]和["<<stk2.top()<<"]是两棵不同子树的根"<<endl;
        return 0;
    }
    while(!stk1.empty() && !stk2.empty() && stk1.top()==stk2.top()){
        stk1.pop();
        stk2.pop();
    }
    int dist=stk1.size()+stk2.size();
    double result=alpha/(alpha+dist);
    cout<<result<<endl;
    return result;
}
 
//计算两个基本关系义原的相似度
double calSimReal(string sem1,string sem2){
    cout<<"将要计算关系义原["<<sem1<<"]和["<<sem2<<"]的相似度"<<endl;
    //如果整体是括在小括号里的，先把小括号去掉
    if(sem1[0]==40){
        sem1.erase(0,1);
        sem1.erase(sem1.size()-1,1);
    }
    if(sem2[0]==40){
        sem2.erase(0,1);
        sem2.erase(sem2.size()-1,1);
    }
    string::size_type p1=sem1.find("=");
    string rela1=sem1.substr(0,p1);
    string::size_type p2=sem2.find("=");
    string rela2=sem2.substr(0,p2);
    if(rela1==rela2){
        string base1=sem1.substr(p1+1);
        string base2=sem1.substr(p2+1);
        return calSimBase(base1,base2);
    }
    else
        return 0;
}
 
//计算第一独立义原描述式的相似度
double calSim1(string line1,string line2){
    if(line1=="" || line2=="")
        return 0;
    cout<<"将要计算第一独立义原描述式["<<line1<<"]和["<<line2<<"]的相似度"<<endl;
    vector<string> vec1,vec2;
    splitString(line1,vec1);
    splitString(line2,vec2);
    assert(vec1.size()==1 && vec2.size()==1);
    return calSimBase(vec1[0],vec2[0]);
}
 
//计算其他独立义原描述式的相似度
double calSim2(string line1,string line2){
    if(line1=="" || line2=="")
        return 0;
    cout<<"将要计算其他独立义原描述式["<<line1<<"]和["<<line2<<"]的相似度"<<endl;
    vector<double> maxsim_vec;
    vector<string> vec1,vec2;
    splitString(line1,vec1);
    splitString(line2,vec2);
 
    int len1=vec1.size();
    int len2=vec2.size();
    while(len1 && len2){
        int m,n;
        double max_sim=0.0;
        for(int i=0;i<len1;++i){
            for(int j=0;j<len2;++j){
                double simil=calSimBase(vec1[i],vec2[j]);
                if(simil>max_sim){
                    m=i;
                    n=j;
                    max_sim=simil;
                }
            }
        }
        if(max_sim==0.0)
            break;
        maxsim_vec.push_back(max_sim);
        vec1.erase(vec1.begin()+m);
        vec2.erase(vec2.begin()+m);
        len1=vec1.size();
        len2=vec2.size();
    }
    //把整体相似度还原为部分相似度的加权平均,这里权值取一样，即计算算术平均
    if(maxsim_vec.size()==0)
        return 0.0;
    double sum=0.0;
    vector<double>::const_iterator itr=maxsim_vec.begin();
    while(itr!=maxsim_vec.end())
        sum+=*itr++;
    return sum/maxsim_vec.size();
}
 
//计算关系义原描述式的相似度
double calSim3(string line1,string line2){
    if(line1=="" || line2=="")
        return 0;
    cout<<"将要计算关系义原描述式["<<line1<<"]和["<<line2<<"]的相似度"<<endl;
    vector<double> sim_vec;
    vector<string> vec1,vec2;
    splitString(line1,vec1);
    splitString(line2,vec2);
     
    int len1=vec1.size();
    int len2=vec2.size();
    while(len1 && len2){
        for(int j=0;j<len2;++j){
            double ss=calSimReal(vec1[len1-1],vec2[j]);
            if(ss!=0){
                sim_vec.push_back(ss);
                vec2.erase(vec2.begin()+j);
                break;  
            }
        }
        vec1.pop_back();
        len1=vec1.size();
        len2=vec2.size();
    }
    if(sim_vec.size()==0)
        return 0.0;
    double sum=0.0;
    vector<double>::const_iterator itr=sim_vec.begin();
    while(itr!=sim_vec.end())
        sum+=*itr++;
    return sum/sim_vec.size();
}
 
//计算符号义原描述式的相似度
double calSim4(string line1,string line2){
    if(line1=="" || line2=="")
        return 0;
    cout<<"将要计算符号义原描述式["<<line1<<"]和["<<line2<<"]的相似度"<<endl;
    vector<double> sim_vec;
    vector<string> vec1,vec2;
    splitString(line1,vec1);
    splitString(line2,vec2);
     
    int len1=vec1.size();
    int len2=vec2.size();
    while(len1 && len2){
        char sym1=vec1[len1-1][0];
        for(int j=0;j<len2;++j){
            char sym2=vec2[j][0];
            if(sym1==sym2){
                string base1=vec1[len1-1].substr(1);
                string base2=vec2[j].substr(1);
                sim_vec.push_back(calSimBase(base1,base2));
                vec2.erase(vec2.begin()+j);
                break;  
            }
        }
        vec1.pop_back();
        len1=vec1.size();
        len2=vec2.size();
    }
     
    if(sim_vec.size()==0)
        return 0.0;
    double sum=0.0;
    vector<double>::const_iterator itr=sim_vec.begin();
    while(itr!=sim_vec.end())
        sum+=*itr++;
    return sum/sim_vec.size();
}
 
//计算两个“概念”的相似度
double calConceptSim(string concept1,string concept2){
    cout<<"将要计算概念["<<concept1<<"]和["<<concept2<<"]的相似度"<<endl;
    if(concept1[0]=='{'){   //概念1是虚词
        if(concept2[0]!='{'){   //概念2是实词
            return 0;
        }
        else{       //概念2是虚词
            string sem1=concept1.substr(1,concept1.size()-2);   //去掉"{"和"}"
            string sem2=concept2.substr(1,concept2.size()-2);
            string::size_type p1=sem1.find("=");
            string::size_type p2=sem2.find("=");
            if(p1==string::npos ^ p2==string::npos){    //一个句法义原，一个是关系义原
                return 0;
            }
            else if(p1==string::npos && p2==string::npos){      //都是句法义原
                return calSimBase(sem1,sem2);
            }
            else{       //都是关系义原
                return calSimReal(sem1,sem2);
            }
        }
    }
    else{       //概念1是实词
        if(concept2[0]=='{'){   //概念2是虚词
            return 0;
        }
        else{       //概念2是实词
            double sim1=0.0;        //分别计算4种描述式的相似度
            double sim2=0.0;
            double sim3=0.0;
            double sim4=0.0;
            string::size_type pos11,pos12,pos21,pos22;
            pos11=pos21=0;
            for(int i=0;i<4;++i){
                pos12=concept1.find("\n",pos11);
                pos22=concept2.find("\n",pos21);
                string sem1=concept1.substr(pos11,pos12-pos11);
                string sem2=concept2.substr(pos21,pos22-pos21);
                switch(i){
                    case 0:
                        sim1=calSim1(sem1,sem2);
                        break;
                    case 1:
                        sim2=calSim2(sem1,sem2);
                        break;
                    case 2:
                        sim3=calSim3(sem1,sem2);
                        break;
                    case 3:
                        sim4=calSim4(sem1,sem2);
                        break;
                    default:
                        break;
                }
                pos11=pos12+1;
                pos21=pos22+1;
            }
            //4部分的加权和作不整体的相似度
            return beta1*sim1+
                    beta2*sim1*sim2+
                    beta3*sim1*sim2*sim3+
                    beta4*sim1*sim2*sim3*sim4;
        }   
    }
}
 
//select回调函数
static int select_callback(void *output_arg,int argc,char *argv[],char *azColName[]){
    vector<string> *vec=(vector<string> *)output_arg;
    string rect(argv[0]);
    vec->push_back(rect);
    return 0;
}
 
//计算两个词语的相似度
double calWordSim(string word1,string word2,sqlite3 *db){
    cout<<"将要计算词语["<<word1<<"]和["<<word2<<"]的相似度"<<endl;
    char *zErrMsg=0;
    int rc;
    vector<string> vec1,vec2;     //两个词语的概念分别存放在vec1和vec2中
    char sql[100]={0};
    sprintf(sql,"select semset from t_gloss where concept=\'%s\'",word1.c_str());
    rc=sqlite3_exec(db,sql,select_callback,&vec1,&zErrMsg);
    assert(rc==SQLITE_OK);
    sprintf(sql,"select semset from t_gloss where concept=\'%s\'",word2.c_str());
    rc=sqlite3_exec(db,sql,select_callback,&vec2,&zErrMsg);
    assert(rc==SQLITE_OK);
     
    int len1=vec1.size();
    int len2=vec2.size();
    if(len1==0)
        cout<<word1<<"不在词典中"<<endl;
    if(len2==0)
        cout<<word2<<"不在词典中"<<endl;
    double maxsim=0.0;
    for(int i=0;i<len1;++i){
        for(int j=0;j<len2;++j){
            double sim=calConceptSim(vec1[i],vec2[j]);
            if(sim>maxsim)
                maxsim=sim;
        }
    }
    return maxsim;
}
 
int main(int argc,char *argv[]){
    if(argc<3){
        cerr<<"Usage:command word1 word2."<<endl;
        return 0;
    }
    string fn("whole.dat");
    initSemVec(fn);
    sqlite3 *db;
    char *zErrMsg=0;
    int rc;
    rc=sqlite3_open("glossary.db",&db);     //打开数据库
    assert(rc==SQLITE_OK);
    string word1(argv[1]);
    string word2(argv[2]);
    double sim=calWordSim(word1,word2,db);
    cout<<"["<<word1<<"]和["<<word2<<"]的相似度是"<<sim<<endl;
    sqlite3_close(db);
    return 0;
}
